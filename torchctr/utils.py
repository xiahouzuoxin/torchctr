import os
from functools import wraps
import pickle
import hashlib
import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # prevent duplicate logs
    logger.propagate = False
    # Prevent duplicate handlers
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = get_logger("torchctr")

def auto_generate_feature_configs(
        df: pd.DataFrame, 
        columns: list = None,
        min_emb_dim: int = 6,
        max_emb_dim: int = 30, 
        max_hash_buckets: int = 1000000,
        seq_max_len: int = 256
    ):
    feat_configs = []

    if columns is None:
        columns = df.columns
    
    for col in columns:
        col_info = {"name": col}
        
        # Check if column contains sequences (lists)
        if df[col].apply(lambda x: isinstance(x, list)).any():
            col_info["dtype"] = "category"
            col_info["islist"] = True
            unique_values = set(val for sublist in df[col] for val in sublist)
            num_unique = len(unique_values)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_info["dtype"] = "numerical"
            col_info["norm"] = "std"  # Standard normalization
            col_info["mean"] = df[col].mean()
            col_info["std"] = df[col].std()
            feat_configs.append(col_info)
            continue
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            col_info["dtype"] = "category"
            unique_values = df[col].unique()
            num_unique = len(unique_values)
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

        # If islist features too long, set max_len to truncate
        if col_info.get("islist", False):
            max_len = max(len(x) for x in df[col])
            col_info["max_len"] = min(max_len, seq_max_len)
        
        # Add the column info to feature configs
        feat_configs.append(col_info)
    
    return feat_configs

def pad_list(arr_list, padding_value, max_len=None):
    '''
    arr_list: list/array of np.array
    '''
    if max_len is None:
        max_len = max([len(arr) for arr in arr_list])

    for k, arr in enumerate(arr_list):
        if len(arr) < max_len:
            arr_list[k] = np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=padding_value)
        else:
            arr_list[k] = np.array(arr[:max_len])
    return arr_list

def jsonify(obj):
    def encoder(obj):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.isnan(obj):
            return None
        raise TypeError(repr(obj) + " is not JSON serializable")
    return json.loads(json.dumps(obj, default=encoder))

def disk_cache(cache_dir='.cache'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique cache key based on the function name and arguments
            cache_key = hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if the result is already cached on disk
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            # Compute the result and save it to disk
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        
        return wrapper
    
    return decorator

def traintest_split(df, test_size=0.2, shuffle=True, group_id=None, random_state=0):
    if group_id is None:
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)
    else:
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state)
        split = splitter.split(df, groups=df[group_id])
        train_inds, test_inds = next(split)
        
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)

        train_df = df.iloc[train_inds]
        test_df = df.iloc[test_inds]
    return train_df, test_df

def traintest_split_by_date(df, date_col, test_size=0.2, shuffle=True, random_state=0):
    uniq_dates = df[date_col].unique()
    uniq_dates = np.sort(uniq_dates)
    n_train_dates = int(len(uniq_dates) * (1 - test_size)) + 1
    assert n_train_dates < len(uniq_dates)
    split_pos = uniq_dates[n_train_dates]
    logger.info(f'Train set date range [{uniq_dates[0]}, {split_pos}]')
    logger.info(f'Test  set date range ({split_pos}, {uniq_dates[-1]}]')
    
    train_df = df[df[date_col] <= split_pos]
    test_df  = df[df[date_col] >  split_pos]
    if shuffle:
        train_df = train_df.sample(frac=1)
    return train_df, test_df
