import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import logger, get_logger

try:
    from accelerate import Accelerator
    logger.info('accelerate is installed.')
    has_accelerate = True
except ImportError:
    logger.warning('accelerate is not installed.')
    has_accelerate = False

class Trainer:
    def __init__(self, model: nn.Module, 
                 optimizer: optim.Optimizer = None, 
                 lr_scheduler=None, 
                 max_epochs=1, 
                 early_stopping_rounds=None, 
                 save_ckpt_path=None,
                 save_ckpt_steps='epoch',
                 ckpt_file_prefix='checkpoint',
                 logger=logger,
                 log_steps=100,
                 training_step_func = None,
                 validation_step_func = None,
                 callback_train_epoch_end = None,
                 callback_eval_epoch_end = None,
                 use_accelerate=has_accelerate,
                 **kwargs):
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The neural network model.
            optimizer (optim.Optimizer, optional): The optimizer for training the model. If not provided, the model's `configure_optimizers` method will be used to configure the optimizer. Defaults to None.
            lr_scheduler (optional): The scheduler for the optimizer. If not provided and the model has a `configure_lr_scheduler` method, it will be used to configure the scheduler. Defaults to None.
            max_epochs (int, optional): The max number of training epochs. Defaults to 1.
            early_stopping_rounds (int, optional): The number of rounds to wait for early stopping. Defaults to None.
            save_ckpt_path (str, optional): The path to save the checkpoint files. Defaults to None.
            save_ckpt_steps (str, optional): The steps to save the checkpoint files. Can be 'epoch' or int value. Defaults to 'epoch'.
            ckpt_file_prefix (str, optional): The prefix of the checkpoint file name. Defaults to 'checkpoint'.
            logger ([type], optional): The logger object. Defaults to logger.
            training_step_func (batch, batch_idx): The training step function. Defaults to use the model's `training_step` method.
            validation_step_func (batch, batch_idx): The validation step function. Defaults to use the model's `validation_step` method.
            callback_train_epoch_end (list): callback functions to be called at the end of each training epoch. 
                Input is a list returns of the `training_step_func`. It helps to calculate the metrics when returns training_step_func contains predictions.
            callback_eval_epoch_end (list): callback functions to be called at the end of each evaluation epoch. 
                Input is a list returns of the validation_step_func. It helps to calculate the metrics when returns validation_step_func contains predictions.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.log_steps = log_steps if log_steps is not None else 100

        self.ckpt_file_prefix = 'checkpoint' if ckpt_file_prefix is None else ckpt_file_prefix
        self.num_epoch = 0 
        self.global_steps = 0    

        if self.optimizer is None and hasattr(self.model, 'configure_optimizers'):
            self.optimizer = self.model.configure_optimizers()

        if self.lr_scheduler is None and hasattr(self.model, 'configure_lr_scheduler'):
            self.lr_scheduler = self.model.configure_lr_scheduler(self.optimizer)

        self.save_ckpt_path = save_ckpt_path
        if self.save_ckpt_path:
            # if os.path.exists(self.save_ckpt_path):
            #     # rename the existing directory
            #     os.rename(save_ckpt_path, f'{save_ckpt_path.rstrip('/')}.old')
            os.makedirs(self.save_ckpt_path, exist_ok=True)

            self.metadata_fn = f'{self.save_ckpt_path}/metadata.json'

        self.save_ckpt_steps = save_ckpt_steps
        if isinstance(self.save_ckpt_steps, int):
            assert self.save_ckpt_steps > 0, 'save_ckpt_steps should be greater than 0.'

        self.max_epochs = max_epochs
        self.early_stopping_rounds = early_stopping_rounds

        self.training_step_func = training_step_func if training_step_func is not None else self.model.training_step
        self.validation_step_func = validation_step_func if validation_step_func is not None else self.model.validation_step
        self.callback_train_epoch_end = callback_train_epoch_end
        self.callback_eval_epoch_end = callback_eval_epoch_end

        assert self.training_step_func is not None and callable(self.training_step_func), 'callable training_step_func is not provided.'
        # assert self.validation_step_func is not None and callable(self.validation_step_func), 'callable validation_step_func is not provided.'
        assert self.optimizer is not None, 'optimizer is not provided.'
        assert self.callback_train_epoch_end is None or callable(self.callback_train_epoch_end), 'callback_train_epoch_end is not callable.'
        assert self.callback_eval_epoch_end is None or callable(self.callback_eval_epoch_end), 'callback_eval_epoch_end is not callable.'

        # all kwargs are saved as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

        if use_accelerate:
            self.accelerator = Accelerator()
            self.logger.info(f'Accelerate device: {self.accelerator.device}')
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)

            self.logger = get_logger('Trainer', level='INFO', use_accelerate=True)
        else:
            self.accelerator = None

    def collect_loss(self, step_func_ret, accumulates: list=None):
        '''
        Collect the loss from the step function return value.
        The first element of the return value must be the loss tensor when it is a list or tuple.
        '''
        if accumulates is None:
            accumulates = []

        def detach_tensor(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.detach().cpu().numpy()
            return v

        if isinstance(step_func_ret, dict):
            accumulates.append( {n: detach_tensor(v) for n,v in step_func_ret.items()} )
            backable_loss = step_func_ret['loss']
        elif isinstance(step_func_ret, (list, tuple)): 
            accumulates.append( type(step_func_ret)([detach_tensor(v) for v in step_func_ret]) )
            backable_loss = step_func_ret[0]
        else:
            accumulates.append( detach_tensor(step_func_ret) )
            backable_loss = step_func_ret
        
        return backable_loss, accumulates
    
    def getstat_lastn_accumulates(self, accumulates: list, n: int=0, agg='mean'):
        '''
        Get the statistics of the last n elements of the accumulates.
        '''
        lastn = accumulates[-n:] if n > 0 else accumulates
        agg_func = np.sum if agg == 'sum' else np.mean
        if isinstance(lastn[0], dict):
            lastn_agg = {k: agg_func([v[k] for v in lastn]).item() for k in lastn[0].keys()}
            final_loss = lastn_agg['loss']
        elif isinstance(lastn[0], (list, tuple)):
            lastn_agg = type(lastn[0])([agg_func([v[i] for v in lastn]).item() for i in range(len(lastn[0]))])
            final_loss = lastn_agg[0]
        else:
            lastn_agg = agg_func(lastn).item()
            final_loss = lastn_agg
        return final_loss, lastn_agg

    def evaluate_model(self, model, eval_dataloader):
        if eval_dataloader is None or model is None or self.validation_step_func is None:
            return None
        
        with torch.no_grad():
            model.eval()
            eval_loss = []

            for k, batch in enumerate(eval_dataloader):
                val_ret = self.validation_step_func(batch, k)
                _, eval_loss = self.collect_loss(val_ret, eval_loss)

            if self.callback_eval_epoch_end:
                self.callback_eval_epoch_end(eval_loss)
        return eval_loss

    def fit(self, 
            train_dataloader: torch.utils.data.DataLoader,
            eval_dataloader: torch.utils.data.DataLoader = None,
            init_ckpt_path: str = None, 
            init_ckpt_exclude_keys: list | str = None,
            ret_model='final'):
        """
        Trains the model using the provided training dataloader and evaluates it using the optional evaluation dataloader.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for training the model.
            eval_dataloader (torch.utils.data.DataLoader, optional): The dataloader for evaluating the model. Defaults to None.
            init_ckpt_path (str, optional): The path to the initial checkpoint path or the file name. Defaults to None.
            init_ckpt_exclude_keys (list, optional): The keys to exclude from the initial checkpoint file. Defaults to None. If 'all_except_model', only model state_dict will loaded.
            ret_model (str, optional): The type of model to return. Can be 'final' or 'best'. Defaults to 'final'.

        Returns:
            The trained model.

        """
        if self.accelerator:
            train_dataloader, eval_dataloader = self.accelerator.prepare([train_dataloader, eval_dataloader])

        final_eval_losses = []
        
        if init_ckpt_path:
            self.load_ckpt(init_ckpt_path, exclude_keys=init_ckpt_exclude_keys)
            eval_loss = self.evaluate_model(self.model, eval_dataloader)
            final_loss, eval_loss = self.getstat_lastn_accumulates(eval_loss, n=len(eval_dataloader), agg='mean')
            self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_loss}')
            final_eval_losses.append(final_loss)

        while self.num_epoch < self.max_epochs:
            self.num_epoch += 1
            
            self.model.train()
            train_loss = []

            # print the latest learning rate of lr_scheduler
            for k, param_group in enumerate(self.optimizer.param_groups):
                self.logger.info(f'Learning rate of group {k}: {param_group["lr"]}')
        
            # Training 
            for k, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()   # zero the parameter gradients
                train_ret = self.training_step_func(batch, k)

                # accumulate losses and compute gradients
                loss, train_loss = self.collect_loss(train_ret, train_loss)

                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()             # compute gradients
                
                self.optimizer.step()       # adjust parameters based on the calculated gradients 
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.global_steps += 1

                if k % self.log_steps == 0:
                    _, latest_loss = self.getstat_lastn_accumulates(train_loss, n=self.log_steps)
                    self.logger.info(f'[Training] Epoch: {self.num_epoch}/{self.max_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {latest_loss}')

                if isinstance(self.save_ckpt_steps, int) and self.global_steps % self.save_ckpt_steps == 0:
                    _, latest_loss = self.getstat_lastn_accumulates(train_loss, n=self.save_ckpt_steps)
                    self.save_ckpt(self.ckpt_file_prefix, 
                                   local_steps=k+len(train_dataloader)*self.num_epoch-1, 
                                   eval_loss=None, train_loss=latest_loss)
                    
            if self.callback_train_epoch_end:
                self.callback_train_epoch_end(train_loss)

            if self.accelerator:
                self.accelerator.wait_for_everyone()

            eval_loss_batches = self.evaluate_model(self.model, eval_dataloader)
            final_eval_loss, eval_loss = self.getstat_lastn_accumulates(eval_loss_batches, n=len(eval_dataloader), agg='mean')
            self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_loss}')

            if self.save_ckpt_steps == 'epoch':
                self.save_ckpt(self.ckpt_file_prefix, 
                            local_steps=self.num_epoch*len(train_dataloader), 
                            eval_loss=final_eval_loss)

            if self.early_stopping_rounds and eval_loss is not None:
                if len(final_eval_losses) >= self.early_stopping_rounds:
                    eval_loss_his_avg = np.mean( final_eval_losses[-self.early_stopping_rounds:] )
                    if final_eval_loss > eval_loss_his_avg:
                        self.logger.info(f'Early stopping at epoch {self.num_epoch}...')
                        break
            final_eval_losses.append(final_eval_loss)

        if ret_model == 'best':
            self.load_ckpt(self.save_ckpt_path, ret_best=True)

        return self.model
    
    def save_ckpt(self, prefix: str, local_steps: int = None, eval_loss=None, **kwargs):
        '''
        Save the checkpoint file with the given prefix and epoch number.

        Args:
            prefix (str): The prefix of the checkpoint file name.
            epoch (int, optional): The epoch number. Defaults to None.
            local_steps (int, optional): The local steps. Defaults to None.
            eval_loss ([type], optional): The evaluation loss. Defaults to None.
        '''
        if self.save_ckpt_path is None:
            return

        state_dict = {
            'local_steps': local_steps,
            'eval_loss': eval_loss,
            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            **kwargs
        }

        # save all the other serializable attributes in __init__ function if not exist in save_dict
        for k, v in self.__dict__.items():
            # check if the attribute is serializable
            if k not in state_dict and not callable(v) and not k.startswith('_'):
                state_dict[k] = v

        torch.save(state_dict, f'{self.save_ckpt_path}/{prefix}.{self.global_steps:06}.ckpt')

        def _update_metadata():
            # save metadata as json file
            if os.path.exists(self.metadata_fn):
                with open(self.metadata_fn, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            if eval_loss is not None and eval_loss < metadata.get('best_eval_loss', float('inf')):
                metadata['best_eval_loss'] = eval_loss
                metadata['best_eval_global_steps'] = self.global_steps
                metadata['best_ckpt'] = f'{prefix}.{self.global_steps:06}.ckpt'

            metadata.update({
                'global_steps': self.global_steps,
                'last_ckpt': f'{prefix}.{self.global_steps:06}.ckpt',
                'local_steps': local_steps,
                'eval_loss': eval_loss
            })
            
            with open(self.metadata_fn, 'w') as f:
                json.dump(metadata, f, indent=4)

        _update_metadata()

        self.logger.info(f'Checkpoint saved at {self.save_ckpt_path}/{prefix}.{self.global_steps:06}.ckpt')

    def load_ckpt(self, path: str=None, exclude_keys=None, ret_best=False, raise_error=True):
        '''
        Load the checkpoint file with the given path.
        It will overwrite the model, optimizer, lr_scheduler and global_steps.

        Args:
            path (str): The path to the checkpoint file or the directory.
            exclude_keys (list, optional): The keys to exclude from the checkpoint file. Defaults to None.
            ret_best (bool, optional): Whether to return the best checkpoint file. Defaults to False.
            raise_error (bool, optional): Whether to raise error if the checkpoint file is not found. Defaults to True.

        Returns:
            The checkpoint file.
        '''
        if path is None:
            return self.load_ckpt(self.save_ckpt_path, exclude_keys=exclude_keys, ret_best=ret_best, raise_error=False)
        else:
            if os.path.isfile(path):
                ckpt_file = path
            elif os.path.isdir(path) and os.path.exists(self.metadata_fn):
                # get checkpoint file from metadata
                with open(self.metadata_fn, 'r') as f:
                    metadata = json.load(f)
                    ckpt_file = metadata['best_ckpt'] if ret_best else metadata['last_ckpt']
                    ckpt_file = os.path.join(path, ckpt_file)
                    if not os.path.exists(ckpt_file):
                        if raise_error:
                            raise FileNotFoundError(f'Checkpoint file {ckpt_file} not found.')
                        else:
                            return None
            else:
                if raise_error:
                    raise FileNotFoundError(f'Checkpoint file {path} not found.')
                else:
                    return None
        
        ckpt = torch.load(ckpt_file, weights_only=False)

        if exclude_keys == 'all_except_model':
            exclude_keys = set(ckpt.keys()) - set(['model',])
        else:
            exclude_keys = set([]) if exclude_keys is None else set(exclude_keys) 
        
        # load all the serializable attributes in the checkpoint file
        for k, v in ckpt.items():
            if k in exclude_keys:
                continue
            elif k in ['model', ]:
                model = eval(f'self.{k}')
                if model is None:
                    model = v
                    continue

                # check the state_dict consistency
                model.load_state_dict(v.state_dict())
                self.logger.info(f'Loaded {k} state_dict from checkpoint.')

                # load the other attributes in the model
                for mk, mv in v.__dict__.items():
                    if mk not in model.__dict__ or callable(mv) or mk.startswith('_') or mk in ['state_dict',]:
                        continue
                    setattr(model, mk, mv)
                    self.logger.info(f'Loaded {k}.{mk} from checkpoint.')
            elif k in self.__dict__:
                self.__dict__[k] = v
                # print it if v is not long
                str_v = str(v) if len(str(v)) < 100 else (str(v)[:100] + '...')
                self.logger.info(f'Loaded {k} = {str_v} from checkpoint.')

        self.logger.info(f'Checkpoint loaded from {ckpt_file}.')

        return ckpt