import torch
import torch.utils
import datasets # huggingface datasets
from .nn.functional import pad_sequences_to_maxlen

def get_dataloader(ds: datasets.Dataset, 
                    feat_configs, 
                    target_cols, 
                    list_padding_value=-100, 
                    list_padding_maxlen=256,
                    **kwargs):
    '''
    Get DataLoader from parquet files.

    Args:
        ds: hugingface datasets.Dataset object. Check https://huggingface.co/docs/datasets/loading#parquet to load parquet files. 
        feat_configs: list of dict, feature configurations for FeatureTransformer
        target_cols: list of str, target columns
        list_padding_value: int, padding value for list features, not used if no list features
        list_padding_maxlen: int, maximum length for padding list features, not used if no list features
        kwargs: DataLoader parameters

    Returns:
        DataLoader object. The batch data format is tuple(features, labels).
        (features, labels) = (
            {
                'dense_features': tensor, 
                'sparse_feature1': tensor, 
                'sparse_feature2': tensor, 
                'seq_sparse_feature1': tensor,
                ...
            },     # features
            tensor # labels
        )
    '''
    ds = ds.with_format('torch')

    def collate_fn(batch):
        # list of dict to dict of list
        # print(batch)

        batch_dense = []
        batch_sparse = {}
        for k in feat_configs:
            if k['type'] == 'dense':
                # print(k['name'])
                batch_dense.append(
                    torch.tensor([sample[k['name']] for sample in batch], dtype=torch.float32)
                )
            elif k['type'] == 'sparse':
                if k.get('islist'):
                    sparse_feat = [sample[k['name']] for sample in batch]
                    # pad sequences
                    sparse_feat = pad_sequences_to_maxlen(
                        sparse_feat, batch_first=True, 
                        padding_value=list_padding_value, 
                        max_length=list_padding_maxlen)
                else:
                    sparse_feat = torch.tensor([[sample[k['name']]] for sample in batch], dtype=torch.long)
                batch_sparse[k['name']] = sparse_feat

        batch_features = {
            'dense_features': torch.stack(batch_dense, dim=1),
            **batch_sparse
        }
        batch_labels = torch.tensor([[sample[f] for f in target_cols] for sample in batch], dtype=torch.float32)
        
        return batch_features, batch_labels

    dl = torch.utils.data.DataLoader(ds, collate_fn=collate_fn, **kwargs)

    return dl
