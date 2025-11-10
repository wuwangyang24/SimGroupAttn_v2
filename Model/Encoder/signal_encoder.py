import torch
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
        norm_layer: Optional[torch.nn.Module] = None,
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

    def SeperateContext(self, attn_scores: torch.Tensor, context_ratio: float=0.5) -> torch.Tensor:
        """Seperate attention scores into context and non-context parts.
        Args:
            attn_scores: Tensor of shape [B, H, N, N], attention scores.
            context_ratio: float, ratio of context patches to total patches.
        Returns:
            context_mask: Tensor of shape [B, M], binary mask indicating context patches.
            non_context_mask: Tensor of shape [B, N-M], binary mask indicating non-context patches.
        """
        B, N, _ = attn_scores.shape
        # Compute the mean attention score for each node
        mean_scores = attn_scores.mean(dim=-1)  # [B, N]
        # Determine a threshold to separate context and non-context nodes
        threshold = mean_scores.mean(dim=-1, keepdim=True)  # [B, 1]
        # Create binary masks based on the threshold
        context_mask = (mean_scores >= threshold).float()  # [B, N]
        object_mask = (mean_scores < threshold).float()   # [B, N]
        return context_mask, object_mask
