import torch
from torch import Tensor
from Backbone.SimpleViT import SimpleViT
from typing import Optional


class ExperienceEncoder(SimpleViT):
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
        memory_embeddings: Tensor
    ) -> Tensor:
        """
        Forward pass through the ContextEncoder.
        Args:
            memory_embeddings: embeddings from memory, shape (B, M*K, D)
            M: number of selected patches
            K: number of most similar memory patches
        Returns:
            Output embeddings, shape (B, M*K, D) or (B, M*K+1, D)
        """
        cls_token_exists = self.cls_token is not None
        # Prepend CLS token if exists
        if cls_token_exists:
            cls_tokens = self.cls_token.expand(memory_embeddings.size(0), -1, -1)  # (B, 1, D)
            memory_embeddings = torch.cat((cls_tokens, memory_embeddings), dim=1)  # (B, M*K+1, D)
        memory_embeddings = self.pos_drop(memory_embeddings)
        # Forward through transformer blocks
        for blk in self.blocks:
            memory_embeddings = blk(memory_embeddings)
        # Normalize
        memory_embeddings = self.norm(memory_embeddings)
        return memory_embeddings
