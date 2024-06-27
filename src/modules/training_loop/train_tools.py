import os
import torch
import torch.nn.functional as F
import gc

def calculate_start_token_penalty(logits, start_token_idx, lambda_penalty):
    """
    Calculate a penalty for predicting the start token at positions other than the first one.

    Args:
    logits (torch.Tensor): Logits from the model of shape [batch, max_caption_len, vocab_size].
    start_token_idx (int): Index of the start token in the vocabulary.
    lambda_penalty (float): Penalty factor to scale the start token penalty.

    Returns:
    torch.Tensor: The calculated start token penalty.
    """
    # Apply softmax to convert logits to probabilities
    token_probs = F.softmax(logits, dim=-1)

    # Exclude the first position and extract the probabilities assigned to the start token
    start_token_prob = token_probs[:, 1:, start_token_idx]

    # Sum probabilities across all positions and batch items, and average over the batch
    penalty = start_token_prob.sum() / logits.size(0)

    # Scale the penalty
    start_token_penalty = penalty * lambda_penalty

    return start_token_penalty

def clear_memory():
    """
    Clear CUDA cache and run garbage collection to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()

def set_visible_device(device):
    """
    Set the device for the model.

    Args:
    device (str): Device to use.
    """
    
    if device == 'cuda:0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif device == 'cuda:1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''