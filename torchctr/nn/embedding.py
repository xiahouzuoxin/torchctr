import torch
import torch.nn as nn
import torch.nn.init as init
import math
from ..utils import logger

def encode_to_nary(input: torch.Tensor, base: int = 2, bit_width: int = 8, bit_width_action: str = 'error') -> torch.Tensor:
    """
    Convert each element in the input tensor to its N-ary (e.g., binary, ternary) representation.
    
    Parameters:
    - input (torch.Tensor): Input tensor containing unsigned integer values.
    - base (int): The base of the encoding (default is 2 for binary).
    - bit_width (int): Desired length of the output representation.
    - bit_width_action (str): How to handle insufficient bit width for the input values. Options are 'error', 'expand' and 'trunc'.
    
    Returns:
    - torch.Tensor: A tensor where each element is represented as a `bit_width`-length vector of digits in base `N`.

    Example:
    >>> input = torch.tensor([1, 2, 3, 4, 5])
    >>> encode_to_nary(input, base=2, bit_width=4)
    tensor([[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1]])
    """
    input = input.int()
    
    # Calculate minimum required bit width
    max_value = input.max().item()
    min_required_bits = math.ceil(math.log(max_value + 1, base))
    
    # Adjust bit width if necessary
    if min_required_bits > bit_width:
        warn_msg = f"bit_width={bit_width} is insufficient to represent values in base-{base}. Minimum required bit width is {min_required_bits}."
        logger.warning(warn_msg)
        if bit_width_action == 'expand':
            bit_width = min_required_bits
            logger.warning(f"Expanding bit width to {bit_width}.")
        elif bit_width_action == 'trunc':
            logger.warning(f"Truncating values to fit within bit width.")
            input = torch.remainder(input, base**bit_width)
        else:
            raise ValueError(warn_msg + " Set `bit_width_action` to 'expand' or 'trunc' to handle this case.")
    
    # Initialize an empty list to collect digits
    nary_digits = []
    input = input.unsqueeze(-1)  # Ensure shape is [batch_size, num_elements, 1] for consistent slicing
    
    # Loop `bit_width` times to get each base-N digit
    for _ in range(bit_width):
        remainder = torch.remainder(input, base)  # Get the remainder for the current base-N digit
        nary_digits.append(remainder)
        input = torch.div(input, base, rounding_mode='floor')  # Integer division to prepare for next digit
    
    # Stack and reverse along the last dimension to have MSB first
    nary_tensor = torch.cat(nary_digits[::-1], dim=-1)
    
    return nary_tensor

class DynamicEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, 
                 scale_grad_by_freq=False, sparse=False, _weight=None):
        super(DynamicEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, 
                                               norm_type, scale_grad_by_freq, sparse, _weight)

    def _expand_embeddings(self, new_num_embeddings):
        if new_num_embeddings <= self.num_embeddings:
            return
        else:
            # only init the expanded embeddings and keep the original embeddings weights
            new_embeddings = torch.empty(new_num_embeddings - self.num_embeddings, self.embedding_dim, 
                                        dtype=self.weight.dtype, device=self.weight.device)
            init.normal_(new_embeddings, mean=0, std=0.01)
            self.weight = nn.Parameter(torch.cat([self.weight, new_embeddings], dim=0))
            self.num_embeddings = new_num_embeddings

    def forward(self, input):
        if input.numel() == 0:
            raise ValueError("Indices tensor is empty")
        if input.min().item() < 0:
            raise ValueError("Indices contain negative values")
        max_index = input.max().item()
        self._expand_embeddings(max_index + 1)
        return super(DynamicEmbedding, self).forward(input)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_embeddings = state_dict[prefix + 'weight'].size(0)
        self._expand_embeddings(num_embeddings)
        if num_embeddings < self.num_embeddings:
            # load part of the weights from the state_dict as embedding size of state_dict is smaller than current embedding size
            state_dict[prefix + 'weight'] = torch.cat([state_dict[prefix + 'weight'], self.weight[num_embeddings:]], dim=0)
        super(DynamicEmbedding, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class NaryDisEmbedding(nn.Module):
    '''
    N-ary Discrete Embedding for Numerical Feature Representation.
    Papers: 
    1. Ads Recommendation in a Collapsed and Entangled World
    2. Numerical Feature Representation with Hybrid N-ary Encoding

    Example:
    >>> embedding = NaryDisEmbedding(embedding_dim=4, encode_bases=[2, 3], reduction='concat')
    >>> input = torch.tensor([1, 2, 3])
    >>> output = embedding(input)
    [[ 0.0012, -0.0034,  0.0023, -0.0012,  0.0012, -0.0034,  0.0023, -0.0012],
     [ 0.0012, -0.0034,  0.0023, -0.0012,  0.0012, -0.0034,  0.0023, -0.0012],
     [ 0.0012, -0.0034,  0.0023, -0.0012,  0.0012, -0.0034,  0.0023, -0.0012]]
    '''
    def __init__(self, embedding_dim, encode_bases=[2, 3], bit_widths=8, bit_width_action='error', reduction='concat'):
        '''
        Initialize the N-ary Discrete Embedding module.
        Args:
        - embedding_dim (int): The dimensionality of the embedding vectors.
        - encode_bases (list): List of bases for N-ary encoding (default is binary and ternary).
        - bit_width (list | int): Desired length of the N-ary representation. Must have the same length as `encode_bases` if a list.
        - bit_width_action (str): How to handle insufficient bit width for the input values. Options are 'error', 'expand' and 'trunc'.
        - reduction (str): The reduction operation to aggregate embeddings for each base. Options are 'concat', 'mean' and 'sum'.
        '''
        super(NaryDisEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleDict()
        for base in encode_bases:
            assert base >= 2, "Base of N-ary encoding must be greater than or equal to 2."
            self.embeddings[str(base)] = nn.Embedding(base, embedding_dim)
            self.embeddings[str(base)].weight.data.normal_(mean=0, std=0.01)

        self.encode_bases = encode_bases if isinstance(encode_bases, list) else [encode_bases]
        self.bit_widths = bit_widths if isinstance(bit_widths, list) else [bit_widths] * len(self.encode_bases)
        assert len(self.encode_bases) == len(self.bit_widths), "Length of `encode_bases` and `bit_widths` must be the same."

        self.bit_width_action = bit_width_action
        self.reduction = reduction

    def forward(self, input):
        '''
        Args:
        - input (torch.Tensor): Input tensor containing unsigned integer values.
        '''
        nary_embeddings = []
        for base, bit_width in zip(self.encode_bases, self.bit_widths):
            nary = encode_to_nary(input, base=base, bit_width=bit_width, bit_width_action=self.bit_width_action)
            nary_emb = self.embeddings[str(base)](nary)
            # Aggregate embeddings for each base
            nary_emb = nary_emb.sum(dim=-2)
            nary_embeddings.append(nary_emb)

        # Reduce embeddings across bases
        if self.reduction == 'concat':
            return torch.cat(nary_embeddings, dim=-1)
        elif self.reduction == 'mean':
            return torch.mean(torch.stack(nary_embeddings), dim=0)
        elif self.reduction == 'sum':
            return torch.sum(torch.stack(nary_embeddings), dim=0)
        else:
            raise ValueError(f"Invalid reduction operation: {self.reduction}. "
                             f"Choose from 'concat', 'mean' and 'sum'.")