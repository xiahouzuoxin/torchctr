import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .utils import logger, get_logger

try:
    from accelerate import Accelerator
    has_accelerate = True
except ImportError:
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
                 training_step_func = None,
                 validation_step_func = None,
                 callback_train_epoch_end = None,
                 callback_eval_epoch_end = None,
                 use_accelerate=has_accelerate,
                 logger=logger,
                 log_steps=100,
                 tb_writer: SummaryWriter=None,
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
            training_step_func (batch, batch_idx): The training step function. Defaults to use the model's `training_step` method.
            validation_step_func (batch, batch_idx): The validation step function. Defaults to use the model's `validation_step` method.
            callback_train_epoch_end (list): callback functions to be called at the end of each training epoch. 
                Input is a list returns of the `training_step_func`. It helps to calculate the metrics when returns training_step_func contains predictions.
            callback_eval_epoch_end (list): callback functions to be called at the end of each evaluation epoch. 
                Input is a list returns of the validation_step_func. It helps to calculate the metrics when returns validation_step_func contains predictions.
            use_accelerate (bool, optional): Whether to use the accelerate library for distributed training. Defaults to has_accelerate.
            logger (logging.Logger, optional): The logger object. Defaults to logger.
            log_steps (int, optional): The steps to log the training loss. Defaults to 100.
            tb_writer (SummaryWriter, optional): The tensorboard writer object. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.log_steps = log_steps if log_steps is not None else 100

        self.ckpt_file_prefix = 'checkpoint' if ckpt_file_prefix is None else ckpt_file_prefix
        self.num_epoch = 0 
        self.global_steps = 0 # steps including previous training from the loaded checkpoint
        self.local_steps = 0 # steps start from this training

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

        self.tb_writer = tb_writer
        if self.tb_writer == True: # initialize a default tensorboard writer
            self.tb_writer = SummaryWriter()

    def collect_loss(self, cur_step_ret, step_rets: list=None):
        '''
        Collect the loss from the step function return value.
        The first element of the return value must be the loss tensor when it is a list or tuple.

        Args:
            cur_step_ret: The return value of the step function of the current step.
            step_rets (list): The list of step return values, including the historical steps if provided. Defaults to None.

        Returns:
            The loss and the list of step return values with the current step appended.
        '''
        if step_rets is None:
            step_rets = []

        def detach_tensor(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.detach().cpu().numpy()
            return v

        if isinstance(cur_step_ret, dict):
            step_rets.append( {n: detach_tensor(v) for n,v in cur_step_ret.items()} )
            loss = cur_step_ret['loss']
        elif isinstance(cur_step_ret, (list, tuple)): 
            step_rets.append( type(cur_step_ret)([detach_tensor(v) for v in cur_step_ret]) )
            loss = cur_step_ret[0]
        else:
            step_rets.append( detach_tensor(cur_step_ret) )
            loss = cur_step_ret
        
        return loss, step_rets
    
    def stat_tail_steps(self, step_rets: list, n: int=0, agg='mean'):
        '''
        Get the statistics of the last/tail n elements of the step_rets.

        Args:
            step_rets (list): The list of step return values.
            n (int, optional): The number of tail elements to aggregate. Defaults to 0 means all elements.
            agg (str, optional): The aggregation function. Can be 'mean' or 'sum'. Defaults to 'mean'.
        '''
        tail = step_rets[-n:] if n > 0 else step_rets
        agg_func = np.sum if agg == 'sum' else np.mean
        if isinstance(tail[0], dict):
            tail_agg = {k: agg_func([v[k] for v in tail]).item() for k in tail[0].keys()}
            loss = tail_agg['loss']
        elif isinstance(tail[0], (list, tuple)):
            tail_agg = type(tail[0])([agg_func([v[i] for v in tail]).item() for i in range(len(tail[0]))])
            loss = tail_agg[0]
        else:
            tail_agg = agg_func(tail).item()
            loss = tail_agg
        return loss, tail_agg

    def evaluate_model(self, model, eval_dataloader):
        if eval_dataloader is None or model is None or self.validation_step_func is None:
            return None
        
        with torch.no_grad():
            model.eval()
            eval_rets = []

            for k, batch in enumerate(eval_dataloader):
                val_ret = self.validation_step_func(batch, k)
                _, eval_rets = self.collect_loss(val_ret, eval_rets)

            if self.callback_eval_epoch_end:
                self.callback_eval_epoch_end(eval_rets)
        return eval_rets

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
            train_dataloader, eval_dataloader = self.accelerator.prepare(train_dataloader, eval_dataloader)

        eval_losses = []
        
        if init_ckpt_path:
            self.load_ckpt(init_ckpt_path, exclude_keys=init_ckpt_exclude_keys)
            eval_rets = self.evaluate_model(self.model, eval_dataloader)
            eval_loss, eval_stats = self.stat_tail_steps(eval_rets, n=len(eval_dataloader), agg='mean')
            self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_stats}')
            eval_losses.append(eval_loss)

        if self.tb_writer:
            try:
                guess_input = next(iter(train_dataloader))
                if isinstance(guess_input, (list, tuple)):
                    guess_input = guess_input[:-1] # assume the last one is target
                self.tb_writer.add_graph(self.model, guess_input, use_strict_trace=False)
                del guess_input
            except Exception as e:
                self.logger.warning(f'Failed to add graph to tensorboard.')

        while self.num_epoch < self.max_epochs:
            self.num_epoch += 1
            
            self.model.train()
            train_rets = []

            # print the latest learning rate of lr_scheduler
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.logger.info(f'Learning rate of group {k}: {param_group["lr"]}')
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        f'Learning rate/group{i}', param_group['lr'], self.global_steps
                    )
        
            # Training 
            for k, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()   # zero the parameter gradients
                train_ret = self.training_step_func(batch, k)

                # accumulate losses and compute gradients
                loss, train_rets = self.collect_loss(train_ret, train_rets)

                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()             # compute gradients
                
                self.optimizer.step()       # adjust parameters based on the calculated gradients 
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    if self.tb_writer:
                        for i, param_group in enumerate(self.optimizer.param_groups):
                            self.tb_writer.add_scalar(
                                f'Learning rate/group{i}', param_group['lr'], self.global_steps
                            )
                self.global_steps += 1
                self.local_steps += 1

                if k % self.log_steps == 0:
                    _, latest_stats = self.stat_tail_steps(train_rets, n=self.log_steps)
                    self.logger.info(
                        f'[Training] Epoch: {self.num_epoch}/{self.max_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {latest_stats}'
                    )
                    if self.tb_writer:
                        if isinstance(latest_stats, dict):
                            self.tb_writer.add_scalars('Training Loss', latest_stats, self.global_steps)
                        elif isinstance(latest_stats, (list, tuple)):
                            for i, v in enumerate(latest_stats):
                                self.tb_writer.add_scalar(f'Training Loss/{i}', v, self.global_steps)
                        else:
                            self.tb_writer.add_scalar(
                                'Training Loss', latest_stats, self.global_steps
                            )

                if isinstance(self.save_ckpt_steps, int) and self.global_steps % self.save_ckpt_steps == 0:
                    _, latest_stats = self.stat_tail_steps(train_rets, n=self.save_ckpt_steps)
                    self.save_ckpt(self.ckpt_file_prefix, eval_loss=None, train_loss=latest_stats)
                    
            if self.callback_train_epoch_end:
                self.callback_train_epoch_end(train_rets)

            if self.accelerator:
                self.accelerator.wait_for_everyone()

            eval_rets = self.evaluate_model(self.model, eval_dataloader)
            eval_loss, eval_stats = self.stat_tail_steps(eval_rets, n=len(eval_dataloader), agg='mean')
            self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_stats}')
            if self.tb_writer:
                if isinstance(eval_stats, dict):
                    self.tb_writer.add_scalars('Validation Loss', eval_stats, self.global_steps)
                elif isinstance(eval_stats, (list, tuple)):
                    for i, v in enumerate(eval_stats):
                        self.tb_writer.add_scalar(f'Validation Loss/{i}', v, self.global_steps)
                else:
                    self.tb_writer.add_scalar('Validation Loss', eval_stats, self.global_steps)

            if self.save_ckpt_steps == 'epoch':
                self.save_ckpt(self.ckpt_file_prefix, eval_loss=eval_loss)

            if eval_loss is not None:
                if self.early_stopping_rounds and len(eval_losses) >= self.early_stopping_rounds:
                    eval_loss_his_avg = np.mean( eval_losses[-self.early_stopping_rounds:] )
                    if eval_loss > eval_loss_his_avg:
                        self.logger.info(f'Early stopping at epoch {self.num_epoch}...')
                        break
                eval_losses.append(eval_loss)

        if ret_model == 'best':
            self.load_ckpt(self.save_ckpt_path, ret_best=True)

        return self.model
    
    def get_state_dict(self, **kwargs):
        '''
        Get the state_dict of the model, optimizer and lr_scheduler.
        '''
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        # save all the other serializable attributes in __init__ function
        other_states = kwargs
        other_states.update(self.__dict__)
        # save member attributes of the model
        for k, v in self.model.__dict__.items():
            if not callable(v) and not k.startswith('_'):
                other_states['model.' + k] = v
        for k, v in other_states.items():
            if k in state_dict or k in (
                'accelerator', 'logger', 'tb_writer',
                'training_step_func', 'validation_step_func', 
                'callback_train_epoch_end', 'callback_eval_epoch_end'
            ):
                continue
            if hasattr(v, 'state_dict'):
                state_dict[k] = v.state_dict()
            elif not callable(v) and not k.startswith('_'):
                # check if the attribute is serializable
                state_dict[k] = v
        return state_dict

    def save_ckpt(self, prefix: str, eval_loss=None, **kwargs):
        '''
        Save the checkpoint file with the given prefix and epoch number.

        Args:
            prefix (str): The prefix of the checkpoint file name.
            epoch (int, optional): The epoch number. Defaults to None.
            eval_loss ([type], optional): The evaluation loss. Defaults to None.
        '''
        if self.accelerator:
            if not self.accelerator.is_main_process:
                return
            self.accelerator.wait_for_everyone()

        if self.save_ckpt_path is None:
            return

        state_dict = self.get_state_dict(**kwargs)
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
                'local_steps': self.local_steps,
                'last_ckpt': f'{prefix}.{self.global_steps:06}.ckpt',
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
        
        ckpt = torch.load(ckpt_file, weights_only=True)

        exclude_keys = set([]) if exclude_keys is None else set(exclude_keys)
        # load all the serializable attributes in the checkpoint file
        for k, v in ckpt.items():
            if k in exclude_keys:
                continue
            if exclude_keys == 'all_except_model' and (k == 'model' or k.startswith('model.')):
                continue
            elif k in ['model', 'optimizer', 'lr_scheduler']:
                restore = eval(f'self.{k}')
                if restore is None:
                    self.logger.warning(f'{restore} restored failed from checkpoint as it is not initialized in the trainer.')
                self.logger.info(f'Loading {k} state_dict from checkpoint.')
                restore.load_state_dict(v)
            elif k in self.__dict__:
                if hasattr(self.__dict__[k], 'load_state_dict'):
                    self.logger.info(f'Loading {k} state_dict from checkpoint.')
                    self.__dict__[k].load_state_dict(v)
                else:
                    # print it if v is not long
                    str_v = str(v) if len(str(v)) < 100 else (str(v)[:100] + '...')
                    self.logger.info(f'Loading {k} = {str_v} from checkpoint.')                    
                    self.__dict__[k] = v
            elif k.startswith('model.'):
                self.logger.info(f'Loading {k} from checkpoint.')
                setattr(self.model, k[6:], v)

        self.logger.info(f'Checkpoint loaded from {ckpt_file}.')

        return ckpt