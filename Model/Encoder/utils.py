import torch
from torch import Tensor
from typing import Optional

def select_positional_embeddings(
    pos_embed: Tensor,
    selected_idx: Tensor,
    cls_token: bool = True
) -> Tensor:
    """
    Select positional embeddings for specific patches.
    Args:
        pos_embed: Tensor of shape (1, N, D) or (1, N+1, D)
        selected_idx: Tensor of selected patch indices, shape (M,)
        cls_token: Whether CLS token exists in pos_embed
    Returns:
        Tensor of shape (1, M+1, D) if cls_token else (1, M, D)
    """
    if cls_token:
        cls_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, D)
        selected_pos_embed = pos_embed[:, 1:, :].index_select(1, selected_idx)  # (1, M, D)
        return torch.cat((cls_pos_embed, selected_pos_embed), dim=1)  # (1, M+1, D)
    else:
        return pos_embed.index_select(1, selected_idx)  # (1, M, D)