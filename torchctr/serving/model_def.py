import json
import numpy as np
import pandas as pd
import torch

from ..dataset import DataFrameDataset

class BaseServingModel:
    def __init__(self, model: torch.nn.Module | str, feat_configs: list[dict] | str = None):
        '''
        model: model object of torch.nn.Module or path to the model checkpoint
        feat_configs: feature configurations, list of dictionaries or path to the feature configurations
        '''
        if isinstance(model, str):
            self.model = torch.load(model, weights_only=False)
            if model.endswith('.ckpt'):
                self.model = self.model['model']
        else:
            self.model = model
        self.model.eval()

        if feat_configs is None:
            print('Feature configurations not provided. Try geting the feature configurations from the model.')
            self.feat_configs = self.model.feat_configs
        elif isinstance(feat_configs, str):
            with open(feat_configs, 'r') as f:
                self.feat_configs = json.load(f)
        else:
            self.feat_configs = feat_configs

        self.ds_generator = DataFrameDataset

    def predict(self, df):
        '''
        Re-implement this method if needed to predict the output of the model.
        '''
        ds = self.ds_generator(df, self.feat_configs, target_cols=None, is_raw=True, is_train=False, n_jobs=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=len(df), shuffle=False, collate_fn=ds.collate_fn)
        preds = []
        for batch in loader:
            pred = self.model(batch)
            preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return {'prediction': preds.tolist()}
