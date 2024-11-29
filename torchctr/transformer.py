from copy import deepcopy
import itertools
import numpy as np
import polars as pl
from .utils import jsonify, logger
from .utils import hash_bucket as hash_bucket_fn

class FeatureTransformer:
    def __init__(self, 
                 feat_configs=None, 
                 category_min_freq=None,
                 category_upper_lower_sensitive=True,
                 list_padding_value=None,
                 list_padding_maxlen=None,
                 outliers_category=[], 
                 outliers_numerical=[], 
                 verbose=False):
        """
        Feature transforming for both train and test dataset with Polars DataFrame (it's much faster than pandas DataFrame).
        Args:
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8, 'maxlen': 256}, # sequence feature, if maxlen is set, it will truncate the sequence to the maximum length
                ]
                If feat_configs is None or list of column names, it will be auto-generated based on the input DataFrame when calling fit() or fit_transform().
            is_train: bool, whether it's training dataset
            category_min_freq: int, minimum frequency for category features, only effective when is_train=True
            outliers_category: list, outliers for category features
            outliers_numerical: list, outliers for numerical features
            verbose: bool, whether to print the processing details
            n_jobs: int, number of parallel jobs
        """
        self.feat_configs = deepcopy(feat_configs)
        self.category_min_freq = category_min_freq
        self.outliers_category = outliers_category
        self.outliers_numerical = outliers_numerical
        self.category_upper_lower_sensitive = category_upper_lower_sensitive
        self.list_padding_value = list_padding_value
        self.list_padding_maxlen = list_padding_maxlen
        self.verbose = verbose

    @staticmethod
    def autogen_init_feat_configs(df: pl.DataFrame, 
                                  columns: list = None, 
                                  min_emb_dim: int = 6, 
                                  max_emb_dim: int = 32,
                                  max_hash_buckets: int = 1000000, 
                                  seq_max_len: int = 256):
        """
        Automatically generate init feature configurations based on the input DataFrame.
        """
        feat_configs = []
        if columns is None:
            columns = df.columns
        dtypes = df.select(columns).dtypes

        for col, dtype in zip(columns, dtypes):
            col_info = {"name": col}
            
            # Check if column contains sequences (lists)
            if dtype == pl.List:
                col_info["dtype"] = "category"
                col_info["islist"] = True
                stats = df.select(
                    nunique = pl.col(col).explode().n_unique(),
                    maxlen = pl.col(col).list.len().max()
                ).to_dicts()[0]
                num_unique = stats['nunique']
                max_len = stats['maxlen']
                col_info["max_len"] = min(max_len, seq_max_len)
            # elif pd.api.types.is_numeric_dtype(df[col]):
            elif dtype.is_numeric():
                col_info["dtype"] = "numerical"
                col_info["norm"] = "std"  # Standard normalization
                stats = df.select(mean=pl.col(col).mean(), std=pl.col(col).std()).to_dicts()[0]
                col_info.update(stats)
                feat_configs.append(col_info)
                continue
            elif dtype == pl.Utf8 or dtype == pl.Categorical:
                col_info["dtype"] = "category"
                num_unique = df.select(pl.col(col).n_unique()).item()
            else:
                continue
            
            if col_info["dtype"] == "category":
                # Calculate embedding dimension
                # emb_dim = int(np.sqrt(num_unique))
                emb_dim = int(np.log2(num_unique))
                emb_dim = min(max(emb_dim, min_emb_dim), max_emb_dim)  # Example bounds
                col_info["emb_dim"] = emb_dim

                # Use hash bucket for high cardinality categorical features or unique values is high
                if num_unique > 0.2 * len(df) or num_unique > max_hash_buckets:
                    # Use hash bucket for high cardinality categorical features
                    col_info["hash_buckets"] = min(num_unique, max_hash_buckets)
                
                col_info["min_freq"] = 3  # Example minimum frequency
            
            # Add the column info to feature configs
            feat_configs.append(col_info)

        logger.info(f'Auto-generated feature configurations: {feat_configs}')

        return feat_configs

    @staticmethod
    def split(df: pl.DataFrame, group_col: str = None, test_size: float = 0.2, shuffle: bool = True, random_state: int = 3407):
        """
        Split the dataset into train and test datasets.
        
        Args:
            df (pl.DataFrame): The input DataFrame to split.
            group_col (str, optional): Column name for group-based splitting. Default is None.
            test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
            random_state (int, optional): Random seed for reproducibility. Default is 3407.
        
        Returns:
            (pl.DataFrame, pl.DataFrame): Train and test DataFrames.
        """
        if group_col:
            groups = df.select(pl.col(group_col)).unique().to_numpy().flatten()
            if shuffle:
                np.random.seed(random_state)
                np.random.shuffle(groups)
            split_idx = int(len(groups) * (1 - test_size))
            train_groups, test_groups = groups[:split_idx], groups[split_idx:]

            train_df = df.filter(pl.col(group_col).is_in(train_groups))
            test_df = df.filter(pl.col(group_col).is_in(test_groups))
        else:
            if shuffle:
                df = df.sample(fraction=1.0, shuffle=True, with_replacement=False, seed=random_state)
            split_idx = int(df.height * (1 - test_size))
            
            train_df = df.slice(0, split_idx)
            test_df = df.slice(split_idx, None)

        return train_df, test_df

    def _add_index(self, df: pl.DataFrame):
        if '_index' not in df.columns:
            df = df.with_columns(
                pl.arange(0, df.height).alias("_index")  # add index column for convenience
            )
        return df
    
    def _init_feat_configs(self, df: pl.DataFrame):
        if self.feat_configs is None:
            logger.warning('Feature configurations are not provided, auto-generating feature configurations...')
            feat_configs = self.autogen_init_feat_configs(df)
        elif all([isinstance(f, str) for f in self.feat_configs]):
            # input is a list of column names
            logger.warning('Feature configurations only contain column names, auto-generating feature configurations...')
            feat_configs = self.autogen_init_feat_configs(df.select(self.feat_configs))
        else:
            feat_configs = self.feat_configs

        assert all([f['dtype'] in ['category', 'numerical'] for f in feat_configs]), 'Only support category and numerical features'
        return feat_configs 

    def fit(self, df: pl.DataFrame):
        """
        Fit the feature transformer based on the training dataset.
        Can be run it multiple times with multiple datasets to update the feature configurations when cannot load all data into memory at once.
        ```
        ft = FeatureTransformer(feat_configs)
        ft.fit(train_df1) # update feat_configs
        ft.fit(train_df2) # update feat_configs with new data based on the previous feat_configs
        ft.fit(train_df3) # update feat_configs with new data based on the previous feat_configs
        ```

        Args:
            df: pl.DataFrame, training dataset
        Returns:
            self: FeatureTransformer
        """
        self.feat_configs = self._init_feat_configs(df)
        self.df = self._add_index(df)

        # Run the feature fitting in parallel, will update the feat_configs in place
        self.df.select([
            self.process_one(pl.col(f['name']), f, mode='fit').alias(f['name']+'_fit')
            for f in self.feat_configs if not f.get('post_cross', None)
        ])

        post_cross_feats = [f for f in self.feat_configs if f.get('post_cross', None)]
        if post_cross_feats:
            self.df.select([
                self.post_cross_features(f, mode='fit') 
                for f in post_cross_feats
            ])

        self.feat_configs = jsonify(self.feat_configs) # make sure it's serializable
        self.df = None

        return self
    
    def transform(self, df: pl.DataFrame):
        """
        Transform the dataset based on the feature configurations.

        Args:
            df: pl.DataFrame, dataset to transform
        Returns:
            df: pl.DataFrame, transformed dataset
        """
        self.df = self._add_index(df)

        # Run the feature fitting in parallel, will update the feat_configs in place
        df = self.df.with_columns([
            self.process_one(pl.col(f['name']), f, mode='transform').alias(f['name'])
            for f in self.feat_configs if not f.get('post_cross', None)
        ])

        post_cross_feats = [f for f in self.feat_configs if f.get('post_cross', None)]
        if post_cross_feats:
            df = df.with_columns([
                self.post_cross_features(f, mode='transform').alias(f['name'])
                for f in post_cross_feats
            ])

        self.df = None # release the memory
        
        return df

    def fit_transform(self, df: pl.DataFrame):
        """
        Fit and transform the dataset based on the feature configurations.

        Args:
            df: pl.DataFrame, training dataset
        Returns:
            df: pl.DataFrame, transformed dataset
        """
        self.feat_configs = self._init_feat_configs(df)
        self.df = self._add_index(df)

        # Run the feature fitting in parallel, will update the feat_configs in place
        df = self.df.with_columns([
            self.process_one(pl.col(f['name']), f, mode='fit_transform').alias(f['name'])
            for f in self.feat_configs if not f.get('post_cross', None)
        ])

        post_cross_feats = [f for f in self.feat_configs if f.get('post_cross', None)]
        if post_cross_feats:
            df = df.with_columns([
                self.post_cross_features(f, mode='fit_transform').alias(f['name'])
                for f in post_cross_feats
            ])

        self.feat_configs = jsonify(self.feat_configs) # make sure it's serializable

        self.df = None # release the memory
        
        return df
    
    def process_one(self, s, feat_config, mode: str='transform'):
        """
        Transform a single feature based on the feature configuration.
        """
        fname = feat_config['name']
        dtype = feat_config['dtype']
        islist = feat_config.get('islist', None)

        if self.verbose:
            logger.info(f'Processing feature {fname}...')

        if islist:
            ret = self.process_list(feat_config, s, mode)
        elif dtype == 'category':
            ret = self.process_category(feat_config, s, mode)
        elif dtype == 'numerical':
            ret = self.process_numerical(feat_config, s, mode)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        return ret

    def _process_category(self, s, oov: str, outliers = None, upper_lower_sensitive = True):
        ''' Convert a series to a string series, with float/integer values considered.
        '''
        # Attempt to convert strings to integers
        int_s = s.cast(pl.Float64, strict=False).fill_nan(None).cast(pl.UInt32, strict=False)
        
        # Identify values that were successfully converted to integers
        is_int_s = int_s.is_not_null()
        
        # Use conditional logic to perform conversions 
        s = (
            pl.when(is_int_s)
            .then(int_s.cast(pl.Utf8))   # Convert to string
            .otherwise(s)  # Ensure None values are replaced with "0"
        )

        if not upper_lower_sensitive:
            s = s.str.to_lowercase()

        if outliers is not None and len(outliers) > 0:
            if isinstance(outliers, list):
                outliers = {v: oov for v in outliers}
            if not isinstance(outliers, dict):
                raise ValueError("Outliers must be a list or a dictionary")
            if not upper_lower_sensitive:
                outliers = {k.lower(): v for k, v in outliers.items()}

            s = s.replace(outliers)

        s = s.fill_null(oov)

        return s

    def process_category(self, feat_config: dict, s: pl.Expr, mode: str = 'transform'):
        """
        Process category features.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary
        outliers_category = feat_config.get('outliers', self.outliers_category)
        category_upper_lower_sensitive = feat_config.get('upper_lower_sensitive', self.category_upper_lower_sensitive)

        s = self._process_category(s, oov, outliers=outliers_category, upper_lower_sensitive=category_upper_lower_sensitive)

        if mode in ('fit', 'fit_transform'):
            feat_config['type'] = 'sparse'
            feat_config['upper_lower_sensitive'] = category_upper_lower_sensitive
            feat_config['oov'] = oov
            hash_buckets = feat_config.get('hash_buckets', None)
            if hash_buckets:
                assert hash_buckets == 'auto' or isinstance(hash_buckets, int), f'hash_buckets should be an integer or "auto" for feature: {name}'
                if hash_buckets == 'auto':
                    hash_buckets = s.n_unique()
                    hash_buckets = self.df.select(hash_buckets).item() # compute the hash_buckets immediately
                    # Auto choose the experienced hash_buckets for embedding table to avoid hash collision
                    if hash_buckets < 100:
                        hash_buckets *= 10
                    elif hash_buckets < 1000:
                        hash_buckets = max(hash_buckets * 5, 1000)
                    elif hash_buckets < 10000:
                        hash_buckets = max(hash_buckets * 2, 5000)
                    elif hash_buckets < 1000000:
                        hash_buckets = max(hash_buckets, 20000)
                    else:
                        hash_buckets = hash_buckets // 10

                    if self.verbose:
                        logger.info(f'Auto hash category {name} with hash_buckets={hash_buckets}...')
                
                feat_config['hash_buckets'] = hash_buckets
                feat_config['num_embeddings'] = hash_buckets
            else:
                raw_vocab = s.value_counts(name='count')
                raw_vocab: pl.DataFrame = self.df.select(raw_vocab).unnest(name)  # compute the raw_vocab immediately

                # low frequency category filtering
                min_freq = feat_config.get('min_freq', self.category_min_freq)
                feat_config['min_freq'] = min_freq
                if min_freq:
                    raw_vocab = raw_vocab.filter(pl.col('count') >= min_freq)

                if len(feat_config.get('vocab', {})) == 0:
                    feat_config['vocab'] = {}
                    idx = 0
                else:
                    idx = max([v['idx'] for v in feat_config['vocab'].values()])

                # update dynamic vocab (should combine with dynamic embedding module when online training)
                for v, cnt in raw_vocab.rows():
                    if v not in feat_config['vocab']:
                        idx += 1
                        feat_config['vocab'][v] = {'idx': idx, 'cnt': cnt}
                    elif v in feat_config['vocab']:
                        feat_config['cnt'][v]['cnt'] += cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': 0, 'cnt': 0}

                if self.verbose:
                    logger.info(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = idx + 1
            
        if mode == 'fit':
            return s
        
        hash_buckets = feat_config.get('hash_buckets', None)
        if hash_buckets:
            s = s.map_elements(lambda x: hash_bucket_fn(x, hash_buckets), return_dtype=pl.Int32)
        else:
            oov_index = feat_config['vocab'].get(oov)['idx']
            s = s.replace_strict(
                old=pl.Series( feat_config['vocab'].keys() ), 
                new=pl.Series( [v['idx'] for v in feat_config['vocab'].values()] ), 
                default=oov_index, 
                return_dtype=pl.UInt32
            ).fill_null(oov_index)

        return s

    def process_numerical(self, feat_config: dict, s: pl.Expr, mode: str = 'transform'):
        """
        Process numerical features.
        """
        name = feat_config['name']
        bins = feat_config.get('bins', None) or feat_config.get('discret', None)
        normalize = feat_config.get('norm', None) or feat_config.get('normalize', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (bins and normalize), f'`bins` and `norm` cannot be set at the same time: {feat_config}'

        if mode in ('fit', 'fit_transform'):
            feat_config['type'] = 'sparse' if bins else 'dense'

            if normalize:
                # calculate mean, std, min, max
                stats = self.df.select(
                    mean=s.mean(), std=s.std(), 
                    min=s.min(), max=s.max(), 
                    cnt=s.len()
                )
                stats = stats.to_dicts()[0] # only 1 row

                # update feat_config
                his_cnt = feat_config.get('cnt', 0)
                feat_config['cnt'] = stats['cnt'] + his_cnt

                if feat_config.get('mean') is None:
                    feat_config['mean'] = stats['mean']
                else:
                    feat_config['mean'] = (feat_config.get('mean') * his_cnt + stats['mean'] * stats['cnt']) / (his_cnt + stats['cnt'])

                if feat_config.get('std') is None:
                    feat_config['std'] = stats['std']
                else:
                    feat_config['std'] = np.sqrt((
                        his_cnt * (feat_config['std'] ** 2) + stats['cnt'] * (stats['std'] ** 2) + 
                        his_cnt * (feat_config['mean'] - stats['mean']) ** 2
                    ) / (his_cnt + stats['cnt']) )

                if feat_config.get('min') is None:
                    feat_config['min'] = stats['min']
                else:
                    feat_config['min'] = min(feat_config.get('min', stats['min']), stats['min'])

                if feat_config.get('max') is None:
                    feat_config['max'] = stats['max']
                else:
                    feat_config['max'] = max(feat_config.get('max', stats['max']), stats['max'])
                    
                if self.verbose:
                    logger.info(f'Feature {name} updated: mean={feat_config["mean"]}, std={feat_config["std"]}, min={feat_config["min"]}, max={feat_config["max"]}')
            elif bins:
                if isinstance(bins, int):
                    nbins = bins
                    qs = np.linspace(0, 1, num=nbins)[1:-1] # exclude 0 and 1 as there are min and max
                    qs_names = [f'{name}_q{int(q*100)}' for q in qs]
                    bins = [s.quantile(q) for q in qs]
                    bins = self.df.select([
                        s.quantile(q).alias(name) for q, name in zip(qs, qs_names)
                    ]).to_numpy().flatten()
                    non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                    feat_config['bins'] = list(bins[non_adjacent_duplicates])
                assert isinstance(bins, list) and len(bins) > 0, f'Invalid bins: {bins} for feature {name}'
                feat_config['num_embeddings'] = len(feat_config['bins']) + 1

        if mode == 'fit':
            return s
        
        if normalize:
            oov = feat_config.get('oov', feat_config['mean'])
            if oov == 'mean':
                oov = feat_config['mean']
            elif oov == 'min':
                oov = feat_config['min']
            elif oov == 'max':
                oov = feat_config['max']
            else:
                oov = float(oov)
            s = s.fill_null(oov).fill_nan(oov)

            if normalize == 'std':
                s = (s - feat_config['mean']) / feat_config['std']
            elif normalize == '[0,1]':
                s = (s - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif bins:
            labels = [str(i) for i in range(feat_config['num_embeddings'])]
            s = s.cut(feat_config['bins'], labels=labels).cast(pl.UInt32)

        return s

    def process_list(self, feat_config, s: pl.Expr, mode: str = 'transform'):
        """
        Process list features.
        """
        name = feat_config['name']

        # if column is string type, split by comma, make sure no space between comma
        if self.df[name].dtype == pl.String:
            if self.verbose:
                logger.info(f'Feature {feat_config["name"]} is a list feature but input string type, split it by comma...')
            s = s.str.split(',').cast(pl.List)
        
        max_len = feat_config.get('maxlen', self.list_padding_maxlen)
        if max_len:
            s = s.list.slice(0, max_len)

        if mode == 'fit':
            flat_s  = s.explode()
        else:
            temp_df = pl.DataFrame({"index": self.df['_index'], name: self.df.select(s)})
            temp_df = temp_df.explode(name)
            flat_s = temp_df.select(pl.col(name)).to_series()

        dtype = feat_config['dtype']
        if dtype == 'category':
            flat_s = self.process_category(feat_config, flat_s, mode)
        elif dtype == 'numerical':
            flat_s = self.process_numerical(feat_config, flat_s, mode)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        if mode == 'fit':
            return s
        
        # group by index and aggregate back
        temp_df = temp_df.with_columns(flat_s.alias("processed"))

        l = s.list.len()
        s = temp_df.group_by("index").agg(pl.col('processed')).sort('index').select("processed").to_series()
        s = pl.when(l > 0).then(s).otherwise([]) # fill empty list with empty list, default is [null] after groupby

        # padding
        padding_value = feat_config.get('padding_value', self.list_padding_value)
        if padding_value and dtype == 'category':
            max_len = max_len or s.len().max()
            df = pl.DataFrame({"original": s})
            df = df.with_columns(pad_list = [0] * (max_len - pl.col('original').len()))
            s = df.select(pl.col('original').list.concat(pl.col('pad_list'))).to_series()
        
        return s

    def post_cross_features(self, feat_config, mode: str = 'transform'):
        """
        Cross features after the single feature processing.
        Only support category features processed by vocabulary.
        """
        name = feat_config['name']
        cross = feat_config.get('post_cross', [])
        assert isinstance(cross, list) and len(cross) > 1, f'cross features: {cross} should be a list with length > 1'

        if self.verbose:
            logger.info(f'Processing post cross features for {name}...')

        cross_feat_configs = []
        for c in cross:
            find = False
            for f in self.feat_configs:
                if f['name'] == c:
                    cross_feat_configs.append(f)
                    find = True
                    break
            assert find, f'cross feature {c} not found'

        oov = feat_config.get('oov', 'other')  # out of vocabulary
        if mode in ('fit', 'fit_transform'):
            cross_vocab = itertools.product(*[zip(f['vocab'].values(), f['vocab'].keys()) for f in cross_feat_configs])

            if len(feat_config.get('vocab', {})) == 0:
                feat_config['vocab'] = {}
                idx = 0
            else:
                idx = max([v['idx'] for v in feat_config['vocab'].values()])
            
            for _, v in enumerate(cross_vocab):
                sources = [_v[1] for _v in v]
                key = [_v[0]['idx'] for _v in v]
                key = '_'.join(map(str, key))
                if key not in feat_config['vocab']:
                    idx += 1
                    feat_config['vocab'][key] = {'idx': idx, 'sources': sources}
            
            if oov not in feat_config['vocab']:
                feat_config['vocab'][oov] = {'idx': 0, 'sources': oov}

            feat_config['num_embeddings'] = idx + 1

            if self.verbose:
                logger.info(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

        if mode == 'fit':
            return feat_config
        
        # cross multiple features
        s = pl.col(cross[0]).cast(pl.Utf8)
        for name in cross[1:]:
            s = s + '_' + pl.col(name).cast(pl.Utf8)

        oov_index = feat_config['vocab'].get(oov)['idx']
        s = s.replace_strict(
            old=pl.Series( feat_config['vocab'].keys() ), 
            new=pl.Series( [v['idx'] for v in feat_config['vocab'].values()] ), 
            default=oov_index, 
            return_dtype=pl.UInt64
        ).fill_null(oov_index)

        return s
    
    def get_feat_configs(self):
        return self.feat_configs
