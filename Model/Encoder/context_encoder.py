import torch
from torch import Tensor
from Backbone.SimpleViT import SimpleViT
from typing import Optional


class ContextEncoder(SimpleViT):
    """ Context Encoder based on SimpleViT architecture."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        representation_size: Optional[int] = None,
        distilled: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[torch.nn.Module] = None,
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
            representation_size=representation_size,
            distilled=distilled,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

    @staticmethod
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

    def forward(
        self, 
        x: Tensor, 
        selected_idx: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass through the ContextEncoder.
        Args:
            x: Input patch embeddings, shape (B, N, D)
            selected_idx: Optional tensor of selected patch indices, shape (M,)
        Returns:
            Output embeddings, shape (B, N_+1, D) or (B, N_, D)
        """
        cls_token_exists = self.cls_token is not None
        # Select positional embeddings first
        if selected_idx is not None:
            pos_embed = self.select_positional_embeddings(
                self.pos_embed, selected_idx, cls_token=cls_token_exists
            )
            x = x[:, selected_idx]  # (B, M, D)
        else:
            pos_embed = self.pos_embed
        # Prepend CLS token if exists
        if cls_token_exists:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, N_+1, D)
        # Add positional embeddings and dropout
        x = x + pos_embed
        x = self.pos_drop(x)
        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # Normalize
        x = self.norm(x)
        return x
