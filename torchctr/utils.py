import pandas as pd
import numpy as np
import json

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
