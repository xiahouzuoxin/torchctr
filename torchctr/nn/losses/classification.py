import torch

def dfm_loss(input: torch.Tensor, target: torch.Tensor, 
                    dfm_logits: torch.Tensor, dfm_elapsed: torch.Tensor, 
                    reduction='mean', weight=None):
    """
    Calculate the loss of Delay Feedback Model (DFM). Mainly used for CVR prediction that there's a delay feedback of label.
    
    Args:
        input (torch.Tensor): The input tensor should be the output of a sigmoid function. for example, pcvr.
        target (torch.Tensor): The target tensor.
        dfm_logits (torch.Tensor): The logits tensor of DFM.
        dfm_elapsed (torch.Tensor): The elapsed time tensor of DFM. 
            Time gap between the click and the conversion for positive samples, 
            and the time gap between the click and the end of the observation window for negative samples.
        reduction (str, optional): The reduction method. Default: 'mean'.
        weight (torch.Tensor, optional): The weight tensor. Default: None.
    
    Paper: Modeling Delayed Feedback in Display Advertising, KDD2014
    """
    lambda_dfm = torch.exp(dfm_logits)
    likelihood_pos = target * (torch.log(input) + dfm_logits - lambda_dfm * dfm_elapsed)
    likelihood_neg = (1 - target) * torch.log( 1 - input + input * torch.exp(-lambda_dfm * dfm_elapsed) )

    likelihood_value = likelihood_pos + likelihood_neg
    if weight is not None:
        likelihood_value = likelihood_value * weight

    if reduction == 'mean':
        return - likelihood_value.mean()
    elif reduction == 'sum':
        return - likelihood_value.sum()
    else:
        raise ValueError('reduction should be either mean or sum')
    