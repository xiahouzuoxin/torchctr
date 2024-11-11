import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def pad_sequences_to_maxlen(sequences, batch_first=False, padding_value=0.0, padding_side='right', max_length=None):
    """
    Pads a list of 1D tensors to the max length of the longest sequence.
    
    Args:
        sequences (list of torch.Tensor): List of 1D tensors with varying lengths.
        batch_first (bool, optional): Whether the output tensor has batch dimension in front. Default is False.
        padding_value (float, optional): The value to use for padding. Default is 0.0.
        padding_side (str, optional): The side to pad on. Default is 'right'.
    
    Returns:
        torch.Tensor: A tensor of shape [batch_size, max_length] with the sequences padded.
    """
    # Pad sequences to the length of the longest sequence in the batch
    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

    if max_length is None:
        return padded_sequences
    
    # Adjust to max_length
    if padded_sequences.size(1) < max_length:
        # Pad further if necessary
        padding_size = max_length - padded_sequences.size(1)
        if padding_side == 'left':
            padded_sequences = torch.nn.functional.pad(
                padded_sequences, (padding_size, 0), mode='constant', value=padding_value
            )
        else:
            padded_sequences = torch.nn.functional.pad(
                padded_sequences, (0, padding_size), mode='constant', value=padding_value
            )
    elif padded_sequences.size(1) > max_length:
        # Truncate if necessary
        if padding_side == 'left':
            padded_sequences = padded_sequences[:, -max_length:]
        else:
            padded_sequences = padded_sequences[:, :max_length]
    
    return padded_sequences

def target_attention(target_emb, candidate_embs, mask=None):
    """
    Compute attention over candidate embeddings with optional masking.

    Args:
        target_emb: Target embedding tensor of shape [B, E].
        candidate_embs: Candidate embeddings tensor of shape [B, N, E].
        mask: Mask tensor of shape [B, N] with 0s indicating positions to mask.

    Returns:
        Weighted sum of candidate embeddings based on attention, tensor of shape [B, E].
    """
    # Compute attention scores
    attn_scores = torch.matmul(candidate_embs, target_emb.unsqueeze(-1)).squeeze(-1)  # [B, N, 1] * [B, 1, E] -> [B, N]

    # Apply mask if provided
    if mask is not None:
        attn_scores.masked_fill(mask == 0, float('-inf'))  # Masked entries are set to -inf to ensure they have no influence on softmax

    # Compute attention weights
    attn_weights = F.softmax(attn_scores, dim=1)

    # Apply attention weights to candidate embeddings
    weighted_embs = attn_weights.unsqueeze(-1) * candidate_embs  # [B, N, 1] * [B, N, E] -> [B, N, E]

    # Sum weighted candidate embeddings
    weighted_emb = weighted_embs.sum(dim=1)  # [B, E]

    return weighted_emb

def epnet(general_embs, domain_embs):
    """
    Compute embeddings for the Embedding Personalized Network (EPNet).

    Args:
        general_embs: List of general embeddings tensors, each of shape [B, E_k]
        domain_embs: List of Domain embeddings tensors, each of shape [B, E_k].

    Returns:
        List of updated general embeddings with domain-specific information
    """
    # Concatenate general and domain embeddings
    # ep_input = torch.cat(domain_embs, dim=-1)

    # Detach general_embs from the computation graph as no gradients are needed
    # general_embs_detached = [emb.detach() for emb in general_embs]
    avg_general_embs = sum(general_embs)   # assume all general embs have the same shape, and this make the epnet smaller
    avg_general_embs = avg_general_embs.detach()

    ep_input = torch.cat(domain_embs + [avg_general_embs, ], dim=-1)

    gates = nn.Sequential(
        nn.Linear(ep_input.shape[-1], ep_input.shape[-1] // 2),
        nn.ReLU(),
        nn.Linear(ep_input.shape[-1] // 2, len(general_embs)),
        nn.Sigmoid()
    )(ep_input)

    gates = gates * 2   # B x len(general_embs)
    gates = gates.split(1, dim=1)

    # Update general embeddings with domain-specific information
    updated_embs = []
    for gate, general_emb in zip(gates, general_embs):
        updated_emb = gate * general_emb
        updated_embs.append(updated_emb)
    
    return updated_embs

