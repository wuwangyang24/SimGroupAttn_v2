import torch
from torch import Tensor
from Backbone.SimpleViT.simpleViT import VisionTransformer
from typing import Optional


class ContextEncoder(VisionTransformer):
    """ Context Encoder based on SimpleViT architecture."""
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
        norm_layer: Optional[torch.nn.Module] = None,
        init_std: float = 0.02,
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
        )

    def forward(
        self, 
        x: Tensor,
    ) -> Tensor:
        """
        Forward pass through the ContextEncoder.
        Args:
            x: Input patch embeddings, shape (B, M, D)
        Returns:
            Output embeddings, shape (B, M, D) or (B, M+1, D)
        """
        cls_token_exists = self.cls_token is not None
        # Prepend CLS token if exists
        if cls_token_exists:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, N_+1, D)
        x = self.pos_drop(x)
        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # Normalize
        x = self.norm(x)
        return x
