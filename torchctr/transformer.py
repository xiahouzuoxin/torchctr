from copy import deepcopy
import numpy as np
import polars as pl
import pandas as pd
from joblib import Parallel, delayed
from .utils import pad_list, jsonify, logger
from .utils import hash_bucket as hash_bucket_fn

class FeatureTransformer:
    def __init__(self, 
                 feat_configs, 
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

        assert all([f['dtype'] in ['category', 'numerical'] for f in self.feat_configs]), 'Only support category and numerical features'

    def _add_index(self, df: pl.DataFrame):
        if '_index' not in df.columns:
            df = df.with_columns(
                pl.arange(0, df.height).alias("_index")  # add index column for convenience
            )
        return df

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
        self.df = self._add_index(df)

        # Run the feature fitting in parallel, will update the feat_configs in place
        self.df.select([
            self.process_one(pl.col(f['name']), f, mode='fit').alias(f['name']+'_fit')
            for f in self.feat_configs
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
            for f in self.feat_configs
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
        self.df = self._add_index(df)

        # Run the feature fitting in parallel, will update the feat_configs in place
        df = self.df.with_columns([
            self.process_one(pl.col(f['name']), f, mode='fit_transform').alias(f['name'])
            for f in self.feat_configs
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
                    feat_config['vocab'][oov] = {'idx': 0, 'freq_cnt': 0}

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
    
    def get_feat_configs(self):
        return self.feat_configs


class FeatureTransformerLegacy:
    def __init__(self, 
                 feat_configs, 
                 category_force_hash=False, 
                 category_dynamic_vocab=True,
                 category_min_freq=None,
                 category_upper_lower_sensitive=True,
                 numerical_update_stats=False,
                 list_padding_value=None,
                 list_padding_maxlen=None,
                 outliers_category=[], 
                 outliers_numerical=[], 
                 verbose=False):
        """
        Feature transforming for both train and test dataset. Deprecated, use FeatureTransformer instead.

        Args:
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8, 'maxlen': 256}, # sequence feature, if maxlen is set, it will truncate the sequence to the maximum length
                ]
            is_train: bool, whether it's training dataset
            category_force_hash: bool, whether to force hash all category features, which will be useful for large category features and online learning scenario, only effective when is_train=True
            category_dynamic_vocab: bool, whether to use dynamic vocab for category features, only effective when is_train=True
            category_min_freq: int, minimum frequency for category features, only effective when is_train=True
            numerical_update_stats: bool, whether to update mean, std, min, max for numerical features, only effective when is_train=True
            outliers_category: list, outliers for category features
            outliers_numerical: list, outliers for numerical features
            verbose: bool, whether to print the processing details
            n_jobs: int, number of parallel jobs
        """
        self.feat_configs = deepcopy(feat_configs)
        self.category_force_hash = category_force_hash
        self.category_dynamic_vocab = category_dynamic_vocab
        self.category_min_freq = category_min_freq
        self.numerical_update_stats = numerical_update_stats
        self.outliers_category = outliers_category
        self.outliers_numerical = outliers_numerical
        self.category_upper_lower_sensitive = category_upper_lower_sensitive
        self.list_padding_value = list_padding_value
        self.list_padding_maxlen = list_padding_maxlen
        self.verbose = verbose

        assert all([f['dtype'] in ['category', 'numerical'] for f in self.feat_configs]), 'Only support category and numerical features'
        assert not (self.category_dynamic_vocab and self.category_force_hash), 'category_dynamic_vocab and category_force_hash cannot be set at the same time'

    def fit(self, df, n_jobs=1):
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
            df: pandas DataFrame
            n_jobs: int, number of parallel jobs
        Returns:
            self: FeatureTransformer
        """
        logger.info('Fitting the feature transformer...')
        self.numerical_update_stats = True
        if not self.category_force_hash:
            self.category_dynamic_vocab = True
        self.fit_transform(df, is_train=True, n_jobs=n_jobs, only_fit=True)
        logger.info('Feature transformer fitted.')
        return self
    
    def transform(self, df, n_jobs=1):
        """
        Transforms the DataFrame based on the feature configurations.
        Args:
            df: pandas DataFrame
            is_train: bool, whether it's training dataset
            n_jobs: int, number of parallel jobs
        Returns:
            df: pandas DataFrame, transformed dataset
        """
        return self.fit_transform(df, is_train=False, n_jobs=n_jobs, only_fit=False)

    def fit_transform(self, df, is_train=True, only_fit=False, n_jobs=1):
        """
        Transforms the DataFrame based on the feature configurations. It's more efficient than calling fit() and transform() separately when is_train=True.
        Args:
            df: pandas DataFrame
        Returns:
            if only_fit=False, return the transformed DataFrame
            if only_fit=True, return self
        """
        if self.verbose:
            logger.info(f'Feature transforming (is_train={is_train}), note that feat_configs will be updated when is_train=True...')
            logger.info(f'Input dataFrame type: {type(df)}, transform it by {self.__class__.__name__}')

        if n_jobs <= 1:
            for k, f in enumerate(self.feat_configs):
                updated_s, updated_f = self._transform_one(df[f['name']], f, is_train, only_fit)
                if is_train:
                    self.feat_configs[k] = updated_f

                if only_fit:
                    # only update feat_configs when only_fit=True
                    continue

                if isinstance(df, pd.DataFrame):
                    df[f['name']] = updated_s
                elif isinstance(df, pl.DataFrame):
                    df = df.with_columns(updated_s.alias(f['name']))

            return self if only_fit else df

        # parallel process features
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_one)(df[f_config['name']], f_config, is_train, only_fit) for f_config in self.feat_configs
        )

        # update df & feat_configs
        for k, (updated_s, updated_f) in zip(range(len(self.feat_configs)), results):
            if is_train:
                self.feat_configs[k] = updated_f

            if only_fit:
                # only update feat_configs when only_fit=True, the updated_s will be None
                continue

            if isinstance(df, pd.DataFrame):
                df[updated_f['name']] = updated_s
            elif isinstance(df, pl.DataFrame):
                df = df.with_columns(updated_s.alias(updated_f['name']))

        # make feat configs json serializable
        self.feat_configs = jsonify(self.feat_configs)

        return self if only_fit else df
    
    def _transform_one(self, s, f, is_train=False, only_fit=False):
        """
        Transform a single feature based on the feature configuration.
        """
        fname = f['name']
        dtype = f['dtype']
        islist = f.get('islist', None)
        pre_transform = f.get('pre_transform', None)

        # pre-process
        if pre_transform:
            if isinstance(s, pd.Series):
                s = s.map(pre_transform)
            elif isinstance(s, pl.Series):
                s = s.map_elements(pre_transform)

        if self.verbose:
            logger.info(f'Processing feature {fname}...')

        if islist:
            updated_s, updated_f = self.process_list(f, s, is_train, only_fit)
        elif dtype == 'category':
            updated_s, updated_f = self.process_category(f, s, is_train, only_fit)
        elif dtype == 'numerical':
            updated_s, updated_f = self.process_numerical(f, s, is_train, only_fit)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        return updated_s, updated_f    

    def update_meanstd(self, s, his_freq_cnt=0, mean=None, std=None):
        """
        Update mean, std for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_mean = s.mean()
        s_std = s.std()

        # update mean and std
        mean = s_mean if mean is None else (mean * his_freq_cnt + s_mean * len(s)) / (his_freq_cnt + len(s))
        std = s_std if std is None else np.sqrt((his_freq_cnt * (std ** 2) + len(s) * (s_std ** 2) + his_freq_cnt * (mean - s_mean) ** 2) / (his_freq_cnt + len(s)))

        return mean, std

    def update_minmax(self, s, min_val=None, max_val=None):
        """
        Update min, max for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_min = s.min()
        s_max = s.max()

        # update min and max
        min_val = s_min if min_val is None else min(min_val, s_min)
        max_val = s_max if max_val is None else max(max_val, s_max)

        return min_val, max_val

    def process_category(self, feat_config: list[dict], s: pd.Series | pl.Series, is_train=False, only_fit=False):
        """
        Process category features.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary

        if isinstance(s, pd.Series):
            input_type = 'pd'
        elif isinstance(s, pl.Series):
            input_type = 'pl'
        else:
            raise ValueError(f'Unsupported data type: {type(s)}')

        outliers_category = feat_config.get('outliers', self.outliers_category)
        if input_type == 'pd':
            s = s.replace(outliers_category, np.nan).fillna(oov).map(lambda x: str(int(x) if type(x) is float else x))
            s = s.astype(str)
        else:
            s = s.map_elements(lambda x: np.nan if x in outliers_category else x, return_dtype=pl.String)
            s = s.fill_null(oov).map_elements(lambda x: str(int(x)) if isinstance(x, float) else str(x), return_dtype=pl.String)

        category_upper_lower_sensitive = feat_config.get('upper_lower_sensitive', self.category_upper_lower_sensitive)
        if not category_upper_lower_sensitive:
            s = s.str.lower() if input_type == 'pd' else s.str.to_lowercase()

        hash_buckets = feat_config.get('hash_buckets')
        if self.category_force_hash and hash_buckets is None:
            hash_buckets = s.nunique() if input_type == 'pd' else s.n_unique()
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
                logger.info(f'Forcing hash category {name} with hash_buckets={hash_buckets}...')

            if is_train:
                feat_config['hash_buckets'] = hash_buckets

        category_dynamic_vocab = feat_config.get('dynamic_vocab', self.category_dynamic_vocab)
        assert not (category_dynamic_vocab and hash_buckets), f'dynamic_vocab and hash_buckets cannot be set at the same time for feature: {name}'

        if is_train:
            # update feat_config
            feat_config['type'] = 'sparse'

            # low frequency category filtering
            raw_vocab = s.value_counts()
            min_freq = feat_config.get('min_freq', self.category_min_freq)
            if min_freq:
                raw_vocab = raw_vocab[raw_vocab >= min_freq] if input_type == 'pd' else raw_vocab.filter(pl.col('count') >= min_freq)

        if hash_buckets:
            if self.verbose:
                logger.info(f'Hashing category {name} with hash_buckets={hash_buckets}...')
            if is_train:
                # update feat_config
                feat_config['num_embeddings'] = hash_buckets
                if min_freq:
                    if input_type == 'pd':
                        feat_config['vocab'] = {v: freq_cnt for v, freq_cnt in raw_vocab.items()}
                    else:
                        feat_config['vocab'] = {row[0]: row[1] for row in raw_vocab.rows()}

            if only_fit:
                return None, feat_config

            if 'vocab' in feat_config:
                if input_type == 'pd':
                    s = s.map(lambda x: x if x in feat_config['vocab'] else oov)
                else:
                    s = s.map_elements(lambda x: x if x in feat_config['vocab'] else oov, return_dtype=pl.String)
                s = s.map(lambda x: x if x in feat_config['vocab'] else oov)
            
            if input_type == 'pd':
                s = s.map(lambda x: hash_bucket_fn(x, hash_buckets)).astype(int)
            else:
                s = s.map_elements(lambda x: hash_bucket_fn(x, hash_buckets), return_dtype=pl.Int32)
        else:
            if self.verbose:
                logger.info(f'Converting category {name} to indices...')
            if is_train:
                if len(feat_config.get('vocab', {})) == 0:
                    feat_config['vocab'] = {}
                    idx = 0
                    category_dynamic_vocab = True  # force dynamic vocab when no vocab is provided
                else:
                    idx = max([v['idx'] for v in feat_config['vocab'].values()])

                # update dynamic vocab (should combine with dynamic embedding module when online training)
                iter_vocab = raw_vocab.items() if input_type == 'pd' else raw_vocab.rows()
                for k, (v, freq_cnt) in enumerate(iter_vocab):
                    if v not in feat_config['vocab'] and category_dynamic_vocab:
                        idx += 1
                        feat_config['vocab'][v] = {'idx': idx, 'freq_cnt': freq_cnt}
                    elif v in feat_config['vocab']:
                        feat_config['vocab'][v]['freq_cnt'] += freq_cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': 0, 'freq_cnt': 0}

                if self.verbose:
                    logger.info(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = idx + 1

            if only_fit:
                return None, feat_config

            # convert to indices
            oov_index = feat_config['vocab'].get(oov)

            if input_type == 'pd':
                s = s.map(lambda x: feat_config['vocab'].get(x, oov_index)['idx']).astype(int)
            else:
                s = s.map_elements(lambda x: feat_config['vocab'].get(x, oov_index)['idx'], return_dtype=pl.Int32)

        return s, feat_config

    def process_numerical(self, feat_config, s, is_train=False, only_fit=False):
        """
        Process numerical features.
        """
        hash_buckets = feat_config.get('hash_buckets', None)
        discretization = feat_config.get('discretization', None) or feat_config.get('discret', None)
        normalize = feat_config.get('norm', None) or feat_config.get('normalize', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'discretization and norm cannot be set at the same time: {feat_config}'

        input_type = 'pd' if isinstance(s, pd.Series) else 'pl'

        if is_train:
            # update mean, std, min, max
            feat_config['type'] = 'sparse' if discretization else 'dense'

            if 'mean' not in feat_config or 'std' not in feat_config or self.numerical_update_stats:
                feat_config['mean'], feat_config['std'] = self.update_meanstd(s, feat_config.get('freq_cnt', 0), mean=feat_config.get('mean'), std=feat_config.get('std'))
                feat_config['freq_cnt'] = feat_config.get('freq_cnt', 0) + len(s)

            if 'min' not in feat_config or 'max' not in feat_config or self.numerical_update_stats:
                feat_config['min'], feat_config['max'] = self.update_minmax(s, min_val=feat_config.get('min'), max_val=feat_config.get('max'))

            if self.verbose:
                logger.info(f'Feature {feat_config["name"]} mean: {feat_config["mean"]}, std: {feat_config["std"]}, min: {feat_config["min"]}, max: {feat_config["max"]}')

            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list(bins[non_adjacent_duplicates])

                feat_config['vocab'] = [np.NaN, float('-inf')] + feat_config['vocab'] + [float('inf')]

        if only_fit:
            return None, feat_config

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = s.fillna(oov) if input_type == 'pd' else s.fill_null(oov)
            s = (s - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = s.fillna(oov) if input_type == 'pd' else s.fill_null(oov)
            s = (s - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            # TODO: support polars
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fillna(0).astype(int)  # index 0 is for nan values

        return s, feat_config

    def process_list(self, feat_config, s, is_train=False, only_fit=False):
        """
        Process list features.
        """
        dtype = feat_config['dtype']
        input_type = 'pd' if isinstance(s, pd.Series) else 'pl'

        # if column is string type, split by comma, make sure no space between comma
        if (input_type == 'pd' and isinstance(s.iat[0], str)) or (input_type == 'pl' and s.dtype == pl.String):
            if self.verbose:
                logger.info(f'Feature {feat_config["name"]} is a list feature but input string type, split it by comma...')
            s = s.str.split(',')
            if dtype == 'numerical':
                if input_type == 'pd':
                    s = s.map(lambda x: [float(v) for v in x if v])
                else:
                    s = s.map_elements(lambda x: [float(v) for v in x if v], return_dtype=pl.List)
        
        max_len = feat_config.get('maxlen', self.list_padding_maxlen)
        if max_len:
            if input_type == 'pd':
                s = s.map(lambda x: x[:max_len] if isinstance(x, list) else x)
            else:
                s = s.map_elements(lambda x: x[:max_len] if isinstance(x, list) else x, return_dtype=pl.List)
        
        if input_type == 'pd':
            flat_s = s.explode()
        else:
            df = pl.DataFrame({"index": list(range(len(s))), "original": s})
            df = df.explode("original")
            flat_s = df["original"]
        
        if dtype == 'category':
            flat_s, updated_f = self.process_category(feat_config, flat_s, is_train, only_fit)
        elif dtype == 'numerical':
            flat_s, updated_f = self.process_numerical(feat_config, flat_s, is_train, only_fit)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')
        
        if only_fit:
            return None, updated_f

        if input_type == 'pd':
            s = flat_s.groupby(level=0).agg(list)
        else:
            df = df.with_columns(flat_s.alias("processed"))
            s = df.group_by("index").agg(pl.col('processed')).sort('index').select("processed").to_series()
        # padding
        padding_value = feat_config.get('padding_value', self.list_padding_value)
        if padding_value and dtype == 'category':
            if input_type == 'pd':
                max_len = min([s.map(len).max(), max_len]) if max_len else s.map(len).max()
                s = s.map(lambda x: pad_list([x], padding_value, max_len)[0])
            else:
                _max_len = s.map_elements(len, return_dtype=pl.Int32).max()
                max_len = min([_max_len, max_len]) if max_len else _max_len
                s = s.map_elements(lambda x: pad_list([x], padding_value, max_len)[0], return_dtype=pl.List)
        return s, updated_f
    
    def get_feat_configs(self):
        return self.feat_configs