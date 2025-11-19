import torch
import torch.nn as nn
from typing import Optional
from Backbone.SimpleViT.simpleViT import VisionTransformer


class SignalEncoder(VisionTransformer):
    """ Signal Encoder based on SimpleViT architecture."""
    def __init__(
        self,
        img_size: list = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
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
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
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

    def forward(self, x: torch.Tensor, return_attn: bool=False) -> torch.Tensor:
        """Seperate attention scores into context and non-context parts. Inspired by https://arxiv.org/pdf/2311.03035v2
        Args:
            x: Tensor of shape [B, 3, W, H], input images.
            attn_scores: Tensor of shape [B, H, N, N], attention scores.
            context_ratio: float, ratio of context patches to total patches.
        Returns:
            non_context_scores: Tensor of shape [B, M], attention scores for non-context patches.
            non_context_patches: Tensor of shape [B, M, D], non-context patch embeddings.
        """
        x, attn_scores = self.forward(x)
        N = attn_scores.shape[-1]
        # Regeneration difficulty
        regeneration = attn_scores.diagonal(dim1=-2, dim2=-1).mean(dim=1) # (B, N)
        # Broadcasting ability
        Broadcasting = (attn_scores.sum(dim=-2) - attn_scores.diagonal(dim1=-2, dim2=-1)).mean(dim=1) # (B, N)
        # Combined score
        combined_score = regeneration * Broadcasting  # (B, N)
        if return_attn:
            return x, combined_score.detach(), attn_scores
        return x, combined_score.detach()
