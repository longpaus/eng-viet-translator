import copy
import torch
import torch.nn.functional as F
from torch import nn


def clones(module: nn.Module, N: int):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def create_causal_mask(seq_len:int):
	"""
	Creates a causal mask for attention scores.

	Args:
		seq_len: The sequence length (L).

	Returns:
		A tensor of shape (L, L) representing the causal mask,
		ready to be used with masked_fill.
	"""
	mask = torch.tril(torch.ones((seq_len, seq_len)))

	return mask.int()


def create_padding_mask(sequences, pad_token:int):
    """
    create a padding mask for attention scores

    Args:
		sequences: A tensor of sequences with shape of (batch_size,L). The each sequence will be padded by the pad_token
		pad_token: The pad token used in sequences

	Returns:
		A tensor of shape (B,1,1,L) and is ready to be used with masked_fill
    """
    mask = (sequences != pad_token).int()
    mask = mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
    return mask


def attention_score(q, k, v, mask:torch.Tensor=None):
    """
    Calculates the scaled dot-product attention scores.

    Args:
        q, k, v: Input tensors of shape (batch_size, h, seq_len, d_k)
        mask: Optional tensor of shape (batch_size, 1, seq_len, seq_len)
    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-1, -2)  # (B, h, L, L)
    scores = scores / torch.sqrt(torch.tensor(d_k))  # Scale by sqrt(d_k)

    # apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)  # Softmax over the sequence length dimension
    output = attention_weights @ v  # (B, h, L, d_k), batch-wise weighted sum of values
    return output
