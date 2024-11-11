import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.utils
from torch.utils.data import Dataset
from .utils import pad_list, logger
from .transformer import FeatureTransformer, FeatureTransformerPolars

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

# Tips: 
# DataFrameDataset is designed for var-length supported dataset, which is suitable for sequence data.
# Another option is to use huggingface's datasets library, which is more powerful and efficient for large-scale datasets.
class DataFrameDataset(Dataset):
    '''
    Var-length supported pytorch dataset for DataFrame.
    '''
    def __init__(self, df: pd.DataFrame | pl.DataFrame, feat_configs: list[dict], target_cols=None, is_raw=True, **kwargs):
        """
        Args:
            df: pandas or polars DataFrame
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8}, # sequence feature
                ]
            target_cols: list of str, target columns
            is_raw: bool, whether the input DataFrame is raw data without feature transforming
            kwargs: include 
                - FeatureTransformer parameters when is_raw=True
                    n_jobs: int, number of parallel jobs
                    category_force_hash: bool, whether to force hash all category features
                    category_upper_lower_sensitive: bool, whether category features are upper/lower case sensitive
                    numerical_update_stats: bool, whether to update mean, std, min, max for numerical features
                    outliers_category: list, outliers for category features
                    outliers_numerical: list, outliers for numerical features
                    verbose: bool, whether to print the processing details
                - list padding parameters
                    list_padding_value: int, padding value for list features, if not set, default to -100
                    list_padding_maxlen: int, maximum length for padding list features, if not set, default to min(256, max_len)
                    list_padding_in_collate_fn: bool, whether to padding in collate_fn, default to True

        Example of using it in DataLoader:
            ```python
            from torch.utils.data import DataLoader
            from torchctr import DataFrameDataset

            ds = DataFrameDataset(df, feat_configs, target_cols, is_raw=True, n_jobs=8)
            DataLoader(ds, collate_fn=ds.collate_fn, batch_size=32, shuffle=True)
            ```
        """
        n_jobs = kwargs.get('n_jobs', 1) # os.cpu_count()
        verbose = kwargs.get('verbose', False)

        self.list_padding_value = kwargs.get('list_padding_value', -100)
        self.list_padding_maxlen = kwargs.get('list_padding_maxlen', 256)
        self.list_padding_in_collate_fn = kwargs.get('list_padding_in_collate_fn', False) # whether to padding in collate_fn
        
        if is_raw:
            assert 'is_train' in kwargs, 'is_train parameter should be provided when is_raw=True'
            is_train = kwargs['is_train']

            if isinstance(df, pd.DataFrame):
                feat_transformer = FeatureTransformer
            elif isinstance(df, pl.DataFrame):
                feat_transformer = FeatureTransformerPolars
            
            self.transformer = feat_transformer(
                feat_configs,
                category_force_hash=kwargs.get('category_force_hash', False),
                category_upper_lower_sensitive=kwargs.get('category_upper_lower_sensitive', True),
                numerical_update_stats=kwargs.get('numerical_update_stats', False),
                list_padding_value=self.list_padding_value if not self.list_padding_in_collate_fn else None,
                list_padding_maxlen=self.list_padding_maxlen if not self.list_padding_in_collate_fn else None,
                outliers_category=kwargs.get('outliers_category', []),
                outliers_numerical=kwargs.get('outliers_numerical', []),
                verbose=verbose
            )
            
            df = self.transformer.transform(df, is_train=is_train, n_jobs=n_jobs)
            feat_configs = self.transformer.get_feat_configs()
            if verbose:
                logger.info(f'Feature transforming (is_train={is_train}) done...')

        self.dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and not f.get('islist')]
        self.seq_dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and f.get('islist')]
        self.sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and not f.get('islist')]
        self.seq_sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and f.get('islist')]
        self.weight_cols_mapping = {f['name']: f.get('weight_col') for f in feat_configs if f['type'] == 'sparse' if f.get('weight')}    
        self.target_cols = target_cols

        self.seq_sparse_configs = {f['name']: f for f in feat_configs if f['type'] == 'sparse' and f.get('islist')}

        if verbose:
            logger.info(f'Dense features: {self.dense_cols}')
            logger.info(f'Sparse features: {self.sparse_cols}')
            logger.info(f'Sequence dense features: {self.seq_dense_cols}')
            logger.info(f'Sequence sparse features: {self.seq_sparse_cols}')
            logger.info(f'Weight columns mapping: {self.weight_cols_mapping}')
            logger.info(f'Target columns: {self.target_cols}')

        self.total_samples = len(df)

        # padding for sequences
        if not is_raw and not self.list_padding_in_collate_fn: # not padded in the transforming process and will not in the collate_fn
            for col in self.seq_sparse_cols:
                list_padding_value = self.seq_sparse_configs[col].get('padding_value', self.list_padding_value)
                list_padding_maxlen = self.seq_sparse_configs[col].get('maxlen', self.list_padding_maxlen)
                list_padding_maxlen = min(
                    list_padding_maxlen,
                    max([len(v) for v in df[col]])
                )

                if isinstance(df, pd.DataFrame):
                    df[col] = pad_list(df[col], list_padding_value, list_padding_maxlen)
                    if col in self.weight_cols_mapping:
                        df[f'{col}_wgt'] = pad_list(df[f'{col}_wgt'], 0., list_padding_maxlen)
                elif isinstance(df, pl.DataFrame):
                    df = df.with_columns(
                        (
                            pl.col(col).map_elements(lambda x: pad_list([x], list_padding_value, list_padding_maxlen)[0], return_dtype=pl.List),
                        ).alias(col)
                    )
                    if col in self.weight_cols_mapping:
                        df = df.with_columns(
                            (
                                pl.col(f'{col}_wgt').map_elements(lambda x: pad_list([x], 0., list_padding_maxlen)[0], return_dtype=pl.List)
                            ).alias(f'{col}_wgt')
                        )

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        self.convert_to_numpy(df)

        if verbose:
            logger.info(f'Finished dataset initialization, total samples: {self.total_samples}')

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        features = {}
        if hasattr(self, 'dense_data'):
            data = self.dense_data[idx]
            # if isinstance(self.dense_data, dd.Series):
            #     data = data.compute()
            features.update( {'dense_features': data} )
        if hasattr(self, 'seq_dense_data'):
            data = self.seq_dense_data[idx]
            # if isinstance(self.seq_dense_data, dd.Series):
            #     data = data.compute()
            features.update( {'seq_dense_features': data} )
        
        all_sparse_data = {**self.sparse_data, **self.seq_sparse_data, **self.weight_data}
        for k,v in all_sparse_data.items():
            features[f'{k}'] = v[idx]

        if hasattr(self, 'target'):
            return features, self.target[idx]
        else:
            return features

    def convert_to_numpy(self, df):
        if self.dense_cols is not None and len(self.dense_cols) > 0:
            self.dense_data = df[self.dense_cols].to_numpy(dtype=np.float32)
        
        if self.seq_dense_cols is not None and len(self.seq_dense_cols) > 0: 
            self.seq_dense_data = df[self.seq_dense_cols].applymap(
                lambda x: np.array([np.nansum(x), np.nanmax(x), np.nanmin(x)])
            ).to_numpy(dtype=np.float32)

        self.weight_data = {}
        self.sparse_data = {}
        for col in self.sparse_cols:
            self.sparse_data[col] = df[[col]].to_numpy(dtype=np.int32)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = df[[weight_col]].to_numpy(dtype=np.float32)  

        # for sparse sequences, padding to the maximum length
        self.seq_sparse_data = {}
        for col, cfg in self.seq_sparse_configs.items():
            # convert to np.array if it's list type
            if not isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].map(np.array)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                if not isinstance(df[weight_col].iloc[0], np.ndarray):
                    df[weight_col] = df[weight_col].map(np.array)

            # return array of arrays
            self.seq_sparse_data[col] = df[col].to_numpy()
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = df[weight_col].to_numpy()

        if self.target_cols is not None:
            self.target = df[self.target_cols].to_numpy(dtype=np.float32)

    def collate_fn(self, batch):
        """
        Collate function for DataFrameDataset.
        It mainly for padding sequences if not padded. If there aren't any sequences, it will call the default collate_fn.
        Args:
            batch: list of tuples, each tuple contains features and target
        Returns:
            batch: list of tensors, each tensor contains features and target
        """
        if not self.list_padding_in_collate_fn:
            return torch.utils.data.dataloader.default_collate(batch)

        if len(batch) == 0:
            return batch
        
        if len(batch[0]) == 2:
            features, target = zip(*batch)
        else:
            features = batch

        batch_feat_keys = set( [k for k in features[0].keys()] )

        # padding for sequences if not padded
        # TODO: how to speed up this process?
        for col in self.seq_sparse_cols:
            wgt_col = f'{col}_wgt'

            list_padding_maxlen = self.seq_sparse_configs[col].get('maxlen', self.list_padding_maxlen)
            list_padding_value = self.seq_sparse_configs[col].get('padding_value', self.list_padding_value)

            max_len = max([len(f[col]) for f in features])
            if max_len <= 1:  # sequence features that length equal to 1 in all samples
                continue
            max_len = min([list_padding_maxlen, max_len]) if list_padding_maxlen else max_len

            # iterate over each sample
            for k, f in enumerate(features):
                updated_f = np.pad(f[col], (0, max_len - len(f[col])), 'constant', constant_values=list_padding_value) if len(f[col]) < max_len else f[col][:max_len]
                # updated_f = pad_list([f[col]], list_padding_value, max_len)
                features[k].update({col: torch.tensor(updated_f, dtype=torch.int)})
                if wgt_col in batch_feat_keys:
                    updated_w = np.pad(f[wgt_col], (0, max_len - len(f[wgt_col])), 'constant', constant_values=0.) if len(f[f'{col}_wgt']) < max_len else f[f'{col}_wgt'][:max_len]
                    # updated_w = pad_list([f[wgt_col]], 0., max_len)
                    features[k].update({wgt_col: torch.tensor(updated_w, dtype=torch.float)})

        # call the original collate_fn
        batch = list(zip(features, target)) if len(batch[0]) == 2 else features
        return torch.utils.data.dataloader.default_collate(batch)

def get_dataloader(ds: DataFrameDataset, **kwargs):
    '''
    Get DataLoader from DataFrameDataset.
    '''
    return torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, **kwargs)
