from copy import deepcopy
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils import murmurhash3_32
from joblib import Parallel, delayed
from .utils import pad_list, jsonify

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

class FeatureTransformer:
    def __init__(self, 
                 feat_configs, 
                 category_force_hash=False, 
                 category_dynamic_vocab=True,
                 category_min_freq=None,
                 category_upper_lower_sensitive=True,
                 numerical_update_stats=False,
                 list_padding_value=None,
                 list_padding_maxlen=None,
                 outliers_category=['','None','none','nan','NaN','NAN','NaT','unknown','Unknown','Other','other','others','Others','REJ','Reject','REJECT','Rejected'], 
                 outliers_numerical=[], 
                 feat_configs_replace=False,
                 verbose=False):
        """
        Feature transforming for both train and test dataset.
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
        self.feat_configs = feat_configs if feat_configs_replace else deepcopy(feat_configs)
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

    def transform(self, df, is_train=False, n_jobs=1):
        """
        Transforms the DataFrame based on the feature configurations.
        Args:
            df: pandas DataFrame
        Returns:
            df: pandas DataFrame, transformed dataset
        """
        if self.verbose:
            print(f'==> Feature transforming (is_train={is_train}), note that feat_configs will be updated when is_train=True...')
            print(f'Input dataFrame type: {type(df)}, transform it by {self.__class__.__name__}')

        if n_jobs <= 1:
            for k, f in enumerate(self.feat_configs):
                updated_s, updated_f = self._transform_one(df[f['name']], f, is_train)
                if isinstance(df, pd.DataFrame):
                    df[f['name']] = updated_s
                elif isinstance(df, pl.DataFrame):
                    df = df.with_columns(updated_s.alias(f['name']))
                if is_train:
                    self.feat_configs[k] = updated_f
            return df

        # parallel process features
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_one)(df[f_config['name']], f_config, is_train) for f_config in self.feat_configs
        )

        # update df & feat_configs
        for k, (updated_s, updated_f) in zip(range(len(self.feat_configs)), results):
            if isinstance(df, pd.DataFrame):
                df[updated_f['name']] = updated_s
            elif isinstance(df, pl.DataFrame):
                df = df.with_columns(updated_s.alias(updated_f['name']))
            if is_train:
                self.feat_configs[k] = updated_f

        # make feat configs json serializable
        self.feat_configs = jsonify(self.feat_configs)

        return df
    
    def _transform_one(self, s, f, is_train=False):
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
            print(f'Processing feature {fname}...')

        if islist:
            updated_s, updated_f = self.process_list(f, s, is_train)
        elif dtype == 'category':
            updated_s, updated_f = self.process_category(f, s, is_train)
        elif dtype == 'numerical':
            updated_s, updated_f = self.process_numerical(f, s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        return updated_s, updated_f    

    def hash(self, v, hash_buckets):
        """
        Hash function for category features.
        """
        # hash_object = hashlib.sha256(str(v).encode())
        # hash_digest = hash_object.hexdigest()
        # hash_integer = int(hash_digest, 16)

        hash_integer = murmurhash3_32(str(v), seed=42, positive=True)
        return hash_integer % hash_buckets

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

    def process_category(self, feat_config, s, is_train=False):
        """
        Process category features.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary

        outliers_category = feat_config.get('outliers', self.outliers_category)
        s = s.replace(outliers_category, np.nan).fillna(oov).map(lambda x: str(int(x) if type(x) is float else x))
        s = s.astype(str)

        category_upper_lower_sensitive = feat_config.get('upper_lower_sensitive', self.category_upper_lower_sensitive)
        if not category_upper_lower_sensitive:
            s = s.str.lower()

        hash_buckets = feat_config.get('hash_buckets')
        if self.category_force_hash and hash_buckets is None:
            hash_buckets = s.nunique()
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
                print(f'Forcing hash category {name} with hash_buckets={hash_buckets}...')

            if is_train:
                feat_config['hash_buckets'] = hash_buckets

        category_dynamic_vocab = feat_config.get('dynamic_vocab', self.category_dynamic_vocab)
        assert not (category_dynamic_vocab and hash_buckets), 'dynamic_vocab and hash_buckets cannot be set at the same time for feature: {name}'

        if is_train:
            # update feat_config
            feat_config['type'] = 'sparse'

            # low frequency category filtering
            raw_vocab = s.value_counts()
            min_freq = feat_config.get('min_freq', self.category_min_freq)
            if min_freq:
                raw_vocab = raw_vocab[raw_vocab >= min_freq]
            
            # check if it's dask series
            # if isinstance(raw_vocab, dd.Series):
            #     raw_vocab = raw_vocab.compute()

        if hash_buckets:
            if self.verbose:
                print(f'Hashing category {name} with hash_buckets={hash_buckets}...')
            if is_train:
                # update feat_config
                feat_config['num_embeddings'] = hash_buckets
                if min_freq:
                    feat_config['vocab'] = {v: freq_cnt for k, (v, freq_cnt) in enumerate(raw_vocab.items())}

            if 'vocab' in feat_config:
                s = s.map(lambda x: x if x in feat_config['vocab'] else oov)
            s = s.map(lambda x: self.hash(x, hash_buckets)).astype(int)
        else:
            if self.verbose:
                print(f'Converting category {name} to indices...')
            if is_train:
                if 'vocab' not in feat_config:
                    feat_config['vocab'] = {}
                    idx = 0

                    category_dynamic_vocab = True # force to dynamic vocab when no vocab is provided
                else:
                    idx = max([v['idx'] for v in feat_config['vocab'].values()])

                # update dynamic vocab (should combine with dynamic embedding module when online training)
                for k, (v, freq_cnt) in enumerate(raw_vocab.items()):
                    if v not in feat_config['vocab'] and category_dynamic_vocab:
                        idx += 1
                        feat_config['vocab'][v] = {'idx': idx, 'freq_cnt': freq_cnt}
                    elif v in feat_config['vocab']:
                        feat_config['vocab'][v]['freq_cnt'] += freq_cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': 0, 'freq_cnt': 0}

                if self.verbose:
                    print(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = idx + 1

            # convert to indices
            oov_index = feat_config['vocab'].get(oov)
            s = s.map(lambda x: feat_config['vocab'].get(x, oov_index)['idx']).astype(int)

        return s, feat_config

    def process_numerical(self, feat_config, s, is_train=False):
        """
        Process numerical features.
        """
        hash_buckets = feat_config.get('hash_buckets', None)
        emb_dim = feat_config.get('emb_dim', None)
        discretization = True if (hash_buckets or emb_dim) else False
        normalize = feat_config.get('norm', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'hash_buckets/emb_dim and norm cannot be set at the same time: {feat_config}'

        if is_train:
            # update mean, std, min, max
            feat_config['type'] = 'sparse' if discretization else 'dense'

            if 'mean' not in feat_config or 'std' not in feat_config or self.numerical_update_stats:
                feat_config['mean'], feat_config['std'] = self.update_meanstd(s, feat_config.get('freq_cnt', 0), mean=feat_config.get('mean'), std=feat_config.get('std'))
                feat_config['freq_cnt'] = feat_config.get('freq_cnt', 0) + len(s)

            if 'min' not in feat_config or 'max' not in feat_config or self.numerical_update_stats:
                feat_config['min'], feat_config['max'] = self.update_minmax(s, min_val=feat_config.get('min'), max_val=feat_config.get('max'))

            if self.verbose:
                print(f'Feature {feat_config["name"]} mean: {feat_config["mean"]}, std: {feat_config["std"]}, min: {feat_config["min"]}, max: {feat_config["max"]}')

            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list(bins[non_adjacent_duplicates])

                feat_config['vocab'] = [np.NaN, float('-inf')] + feat_config['vocab'] + [float('inf')]

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fillna(0).astype(int)  # index 0 is for nan values

        return s, feat_config

    def process_list(self, feat_config, s, is_train=False):
        """
        Process list features.
        """
        dtype = feat_config['dtype']

        # if column is string type, split by comma, make sure no space between comma
        if isinstance(s.iat[0], str):
            if self.verbose:
                print(f'Feature {feat_config["name"]} is a list feature but input string type, split it by comma...')
            s = s.str.split(',')
            if dtype == 'numerical':
                s = s.map(lambda x: [float(v) for v in x if v])
        
        max_len = feat_config.get('maxlen', self.list_padding_maxlen)
        if max_len:
            s = s.map(lambda x: x[:max_len] if isinstance(x, list) else x)
        flat_s = s.explode()
        if dtype == 'category':
            flat_s, updated_f = self.process_category(feat_config, flat_s, is_train)
        elif dtype == 'numerical':
            flat_s, updated_f = self.process_numerical(feat_config, flat_s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')
        s = flat_s.groupby(level=0).agg(list)
        # padding
        padding_value = feat_config.get('padding_value', self.list_padding_value)
        if padding_value and dtype == 'category':
            max_len = min([s.map(len).max(), max_len]) if max_len else s.map(len).max()
            s = s.map(lambda x: pad_list([x], padding_value, max_len))
        return s, updated_f
    
    def get_feat_configs(self):
        return self.feat_configs

class FeatureTransformerPolars(FeatureTransformer):

    def process_category(self, feat_config: list[dict], s: pl.Series, is_train=False):
        """
        Process category features using Polars.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary

        outliers_category = feat_config.get('outliers', self.outliers_category)
        s = s.map_elements(lambda x: np.nan if x in outliers_category else x, return_dtype=pl.String)
        s = s.fill_null(oov).map_elements(lambda x: str(int(x)) if isinstance(x, float) else str(x), return_dtype=pl.String)

        category_upper_lower_sensitive = feat_config.get('upper_lower_sensitive', self.category_upper_lower_sensitive)
        if not category_upper_lower_sensitive:
            s = s.str.to_lowercase()

        hash_buckets = feat_config.get('hash_buckets')
        if self.category_force_hash and hash_buckets is None:
            hash_buckets = s.n_unique()
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
                print(f'Forcing hash category {name} with hash_buckets={hash_buckets}...')

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
                raw_vocab = raw_vocab.filter(pl.col('count') >= min_freq)

        if hash_buckets:
            if self.verbose:
                print(f'Hashing category {name} with hash_buckets={hash_buckets}...')
            if is_train:
                # update feat_config
                feat_config['num_embeddings'] = hash_buckets
                if feat_config.get('min_freq'):
                    feat_config['vocab'] = {row[0]: row[1] for row in raw_vocab.rows()}

            if 'vocab' in feat_config:
                s = s.map_elements(lambda x: x if x in feat_config['vocab'] else oov, return_dtype=pl.String)
            s = s.map_elements(lambda x: self.hash(x, hash_buckets), return_dtype=pl.Int32)
        else:
            if self.verbose:
                print(f'Converting category {name} to indices...')
            if is_train:
                if len(feat_config.get('vocab', {})) == 0:
                    feat_config['vocab'] = {}
                    idx = 0
                    category_dynamic_vocab = True  # force dynamic vocab when no vocab is provided
                else:
                    idx = max([v['idx'] for v in feat_config['vocab'].values()])

                # update dynamic vocab
                for (v, freq_cnt) in raw_vocab.rows():
                    if v not in feat_config['vocab'] and category_dynamic_vocab:
                        idx += 1
                        feat_config['vocab'][v] = {'idx': idx, 'freq_cnt': freq_cnt}
                    elif v in feat_config['vocab']:
                        feat_config['vocab'][v]['freq_cnt'] += freq_cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': 0, 'freq_cnt': 0}

                if self.verbose:
                    print(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = idx + 1

            # convert to indices
            oov_index = feat_config['vocab'].get(oov)
            s = s.map_elements(lambda x: feat_config['vocab'].get(x, oov_index)['idx'], return_dtype=pl.Int32)

        return s, feat_config

    def process_numerical(self, feat_config, s, is_train=False):
        """
        Process numerical features.
        """
        hash_buckets = feat_config.get('hash_buckets', None)
        emb_dim = feat_config.get('emb_dim', None)
        discretization = True if (hash_buckets or emb_dim) else False
        normalize = feat_config.get('norm', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'hash_buckets/emb_dim and norm cannot be set at the same time: {feat_config}'

        if is_train:
            # update mean, std, min, max
            feat_config['type'] = 'sparse' if discretization else 'dense'

            if 'mean' not in feat_config or 'std' not in feat_config or self.numerical_update_stats:
                feat_config['mean'], feat_config['std'] = self.update_meanstd(s, feat_config.get('freq_cnt', 0), mean=feat_config.get('mean'), std=feat_config.get('std'))
                feat_config['freq_cnt'] = feat_config.get('freq_cnt', 0) + len(s)

            if 'min' not in feat_config or 'max' not in feat_config or self.numerical_update_stats:
                feat_config['min'], feat_config['max'] = self.update_minmax(s, min_val=feat_config.get('min'), max_val=feat_config.get('max'))

            if self.verbose:
                print(f'Feature {feat_config["name"]} mean: {feat_config["mean"]}, std: {feat_config["std"]}, min: {feat_config["min"]}, max: {feat_config["max"]}')

            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list(bins[non_adjacent_duplicates])

                feat_config['vocab'] = [np.NaN, float('-inf')] + feat_config['vocab'] + [float('inf')]

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fill_null(oov) - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fill_null(oov) - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            # FIXME: migrate to Polars
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fill_null(0).astype(int)  # index 0 is for nan values

        return s, feat_config

    def process_list(self, feat_config, s, is_train=False):
        """
        Process list features.
        """
        dtype = feat_config['dtype']

        # if column is string type, split by comma, make sure no space between comma
        if s.dtype == pl.String:
            if self.verbose:
                print(f'Feature {feat_config["name"]} is a list feature but input string type, split it by comma...')
            s = s.str.split(',')
            if dtype == 'numerical':
                s = s.map_elements(lambda x: [float(v) for v in x if v], return_dtype=pl.List)
        
        max_len = feat_config.get('maxlen', self.list_padding_maxlen)
        if max_len:
            s = s.map_elements(lambda x: x[:max_len] if isinstance(x, list) else x, return_dtype=pl.List)

        # Explode the list into separate rows
        df = pl.DataFrame({"index": list(range(len(s))), "original": s})
        df = df.explode("original")
        flat_s = df["original"]
        if dtype == 'category':
            flat_s, updated_f = self.process_category(feat_config, flat_s, is_train)
        elif dtype == 'numerical':
            flat_s, updated_f = self.process_numerical(feat_config, flat_s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        df = df.with_columns(flat_s.alias("processed"))
        s = df.group_by("index").agg(pl.col('processed')).sort('index').select("processed").to_series()

        # padding
        padding_value = feat_config.get('padding_value', self.list_padding_value)
        if padding_value and dtype == 'category':
            _max_len = s.map_elements(len, return_dtype=pl.Int32).max()
            max_len = min([_max_len, max_len]) if max_len else _max_len
            s = s.map_elements(lambda x: pad_list([x], padding_value, max_len), return_dtype=pl.List)
        return s, updated_f
