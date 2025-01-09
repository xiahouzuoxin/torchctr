import json
import torch
import pandas as pd
import polars as pl
from torchctr.nn.functional import pad_sequences_to_maxlen
from torchctr.transformer import FeatureTransformer
from torchctr.utils import logger

from ..models import DNN

class ServingDNNModel:
    def __init__(self, ckpt_path: str, feat_configs: list[dict] | str = None):
        '''
        Args:
            ckpt_path: path to the model checkpoint
            feat_configs: feature configurations, list of dictionaries or path to the feature configurations
        '''
        ckpt = torch.load(ckpt_path, weights_only=True)
        if feat_configs is None and 'feat_configs' in ckpt:
            self.feat_configs = ckpt['feat_configs']
        elif isinstance(feat_configs, str):
            with open(feat_configs, 'r') as f:
                self.feat_configs = json.load(f)
        else:
            raise ValueError('feat_configs is required')
        
        self.model = DNN(self.feat_configs, [128,64,32])

        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.ft = FeatureTransformer(self.feat_configs)

    def predict(self, input):
        '''
        Re-implement this method.
        '''

        if isinstance(input, pd.DataFrame):
            input = pl.from_pandas(input)
        df = self.ft.transform(input)

        # convert to torch tensor
        dense_cols = [f['name'] for f in self.feat_configs if f['type'] == 'dense']
        dense_features = torch.tensor(df.select(pl.col(dense_cols)).to_numpy(), dtype=torch.float32)

        features = {
            'dense_features': dense_features
        }
        for k in self.feat_configs:
            if k['type'] != 'sparse':
                continue

            if k.get('islist'):
                # convert to list of torch tensor
                sparse_feat = [torch.tensor(v, dtype=torch.long) for v in df.select(pl.col(k['name'])).to_series().to_numpy()]
                # pad sequences
                sparse_feat = pad_sequences_to_maxlen(sparse_feat, batch_first=True, padding_value=-100, max_length=5)
            else:
                sparse_feat = torch.tensor(df.select(pl.col(k['name'])).to_numpy(), dtype=torch.long)

            features[k['name']] = sparse_feat

        # predict
        with torch.no_grad():
            output = self.model(features)

        if isinstance(output, tuple):
            output = [o.detach().cpu().numpy() for o in output]
        else:
            output = output.detach().cpu().numpy()

        return output.tolist()
        

