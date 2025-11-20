from .Encoder.memory_encoder import MemoryEncoder
from .Encoder.signal_encoder import SignalEncoder
from .Memory.memory_bank import MemoryBank
import torch
import os

    
class MemoryJepa(torch.nn.Module):
    """ MemoryJEPA model combining MemoryEncoder and SignalEncoder."""
    def __init__(self, cfg: dict):
        super().__init__()
        signal_encoder_cfg = cfg.get("signal_encoder", {})
        memory_encoder_cfg = cfg.get("memory_encoder", {})
        memory_bank_cfg = cfg.get("memory_bank", {})
        self.signal_encoder = SignalEncoder(**signal_encoder_cfg)
        self.memory_encoder = MemoryEncoder(**memory_encoder_cfg)
        self.memory_bank = MemoryBank(
            capacity=memory_bank_cfg.get("memory_capacity", 100000),
            embed_dim=memory_bank_cfg.get("embed_dim", 768),
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        )
        self.loss_fn = self._loss_fn(cfg.get("loss_type", "cosine"))

    def _loss_fn(self, loss_type: str) -> any:
        """ Returns loss function based on loss_type."""
        if loss_type == 'cosine':
            cos = torch.nn.CosineSimilarity(dim=-1)
            return lambda x, y: (1 - cos(x, y)).mean()
        elif loss_type == 'mse':
            return torch.nn.MSELoss(reduction=None)
        elif loss_type == 'ce':
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented.")

    def forward(self, 
                x: any, 
                num_neighbors: int=5, 
                remain_signal_ratio: float=0.1, 
                memory_mode: str='random',
                return_attn: bool = False
                ) -> dict:
        """
        Args:
            x: Tensor of shape [B, 3, W, H], input images.
            num_neighbors: number of nearest neighbors to retrieve.
            remain_signal_ratio: float, ratio of original signal to retain.
        Returns:
            embeddings: output embeddings from memory encoder.
            loss: float, loss between cls_signal and cls_memory.
            attn_scores: attention scores from signal encoder
        """
        if return_attn:
            x, combined_scores, attn_scores = self.signal_encoder(x, return_attn=return_attn)  # [B, N, D],  [B, N], [B, H, N, N]
        else:
            x, combined_scores = self.signal_encoder(x)  # [B, N, D],  [B, N]
            attn_scores = None
        # normalize combined scores
        combined_scores = combined_scores / (combined_scores.sum(dim=1, keepdim=True) + 1e-6)
        if self.signal_encoder.cls_token is not None:
            cls_signal, x = x[:,0], x[:,1:]
            combined_scores = combined_scores[:, 1:]
        # update memory bank
        B, N, D = x.shape
        self.memory_bank.memorize(x.reshape(B*N, D), combined_scores.reshape(-1), mode=memory_mode)
        # Encode with memory encoder
        memory_embeddings = self.memory_encoder(x, self.memory_bank, num_neighbors, remain_signal_ratio)  # [B, M, D]
        if self.memory_encoder.cls_token:
            cls_memory, memory_embeddings = memory_embeddings[:, 0], memory_embeddings[:, 1:]
        # calculate loss
        loss = (self.loss_fn(x, memory_embeddings).mean(dim=-1) * combined_scores).mean()
        #loss = (self.loss_fn(x, memory_embeddings).mean(dim=-1)).mean()
        return {'embeddings': memory_embeddings, 'loss': loss, 'attn_scores': attn_scores}