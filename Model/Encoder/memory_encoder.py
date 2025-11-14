import torch
import torch.nn as nn
from torch import Tensor
from Backbone.SimpleViT.simpleViT import VisionTransformer
from typing import Optional


class MemoryEncoder(VisionTransformer):
    """ memory Encoder based on SimpleViT architecture."""
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[torch.nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        cls_token: bool = True,
        return_attention: bool = True
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            cls_token=cls_token,
            return_attention=return_attention
        )
        self.patch_embed = None # Bypass patch embedding
        self.pos_embed = None # Bypass position embedding

    # override
    def forward(self, x: Tensor, memorybank: any, k: int, remain_signal_ratio: float=0.1) -> Tensor:
        """ Forward pass with memory recollection.
        Args:
            x: Tensor of shape [B, M, D], input images.
            k: number of nearest neighbors to retrieve.
            remain_signal_ratio: float, ratio of original signal to retain.
        Returns:
            Tensor of shape [B, M*k, D] or [B, M*k+1, D].
        """
        if k > 0:
            _, memory_embeddings = memorybank.recollect(x, k)  # indices: [B, M*k, D]
            x = x.repeat_interleave(k, dim=1)  # [B, M*k, D]
            x = memory_embeddings * (1 - remain_signal_ratio) + x * remain_signal_ratio  # [B, M*k, D]
        x = super().forward(x)  # [B, M*k, D] or [B, M*k+1, D]
        return x