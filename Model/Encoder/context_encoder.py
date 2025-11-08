import torch
from Backbone.SimpleViT import SimpleViT


class ContextEncoder(SimpleViT):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 representation_size=None, 
                 distilled=False, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=None):
        super(ContextEncoder, self).__init__(img_size=img_size, 
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
                                             norm_layer=norm_layer)        
    
    def select_postional_embeddings(self, pos_embed, selected_idx, cls_token=True):
        # pos_embed: (1, N, D) or (1, N+1, D)
        # selected_idx: (M,) selected patch indices
        if not cls_token:
            selected_pos_embed = pos_embed.index_select(1, selected_idx)  # (1, M, D)
            return selected_pos_embed
        cls_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, D)
        selected_pos_embed = pos_embed[:, 1:, :].index_select(1, selected_idx)  # (1, M, D)
        new_pos_embed = torch.cat((cls_pos_embed, selected_pos_embed), dim=1)  # (1, M+1, D)
        return new_pos_embed
    
    def forward(self, x, selected_idx=None):
        # x: (B, C, H, W)
        # selected_idx: (M,) selected patch indices
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
            x = torch.cat((cls_tokens, x[:, selected_idx]), dim=1)  # (B, N_+1, D)
        if selected_idx is not None:
            pos_embed = self.select_postional_embeddings(self.pos_embed, selected_idx, cls_token=(self.cls_token is not None))
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, N_+1, D) or (B, N_, D)
