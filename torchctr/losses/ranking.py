import torch
import torch.nn.functional as F

def pairwise_loss_with_logits(input: torch.Tensor, target: torch.Tensor, reduction='mean', weight=None):
    """
    Calculate the pairwise rank loss with logits input.

    Args:
        input (torch.Tensor): The input tensor should be the logits value.
        target (torch.Tensor): The target tensor, can be binary or float value.
        reduction (str, optional): The reduction method. Default: 'mean'.
        weight (str, optional): The weight tensor. Default: None.

    Paper: Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model
    Link: https://arxiv.org/abs/2208.06164
    """
    # assert input.dim() > 1 and target.dim() > 1, 'input and target should be 2-dim or higher'
    if input.dim() == 1:
        input = input.unsqueeze(1)
    if target.dim() == 1:
        target = target.unsqueeze(1)

    pairwise_target = target - target.t()
    pairwise_mask = ( pairwise_target > 0 )
    pairwise_input = input - input.t()
    pairwise_input = pairwise_input[pairwise_mask]
    
    if weight is None:
        pairwise_weight = None
    elif isinstance(weight, str):
        if weight == 'linear':
            pairwise_weight = pairwise_target[pairwise_mask]
        elif weight == 'log':
            pairwise_weight = torch.log(pairwise_target[pairwise_mask] + 1)
        elif isinstance(weight, torch.Tensor):
            raise ValueError('weight should be either None, linear or log when it is a string')
    elif isinstance(weight, torch.Tensor):
        assert weight.shape == pairwise_target.shape, 'weight shape should be the same as pairwise_target (target - target.t())'
        pairwise_weight = weight[pairwise_mask]
    else:
        raise ValueError('weight should be either None, linear or log when it is a string or a tensor')

    # if length of pairwise_logits is zero, return zero
    if pairwise_input.numel() == 0:
        return 0.
    
    # calc - y * log(p)
    pairwise_pseudo_labels = torch.ones_like(pairwise_input)
    rank_loss = F.binary_cross_entropy_with_logits(input=pairwise_input, target=pairwise_pseudo_labels, weight=pairwise_weight, reduction=reduction)
    return rank_loss
