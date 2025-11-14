from Encoder.memory_encoder import MemoryEncoder
from Encoder.signal_encoder import SignalEncoder
from Memory.memory_bank import MemoryBank
import torch


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
            capacity=memory_bank_cfg.get("memory_capacity", 10000),
            embed_dim=memory_bank_cfg.get("embed_dim", 768),
            device=memory_bank_cfg.get("memory_device", 'gpu'),
            dtype=memory_bank_cfg.get("memory_dtype", torch.float16)
        )
        self.loss_fn = self._loss_fn(cfg.get("loss_type", "cosine"))

    def _loss_fn(self, loss_type: str) -> any:
        """ Returns loss function based on loss_type."""
        if loss_type == 'cosine':
            cos = torch.nn.CosineSimilarity(dim=-1)
            return lambda x, y: (1 - cos(x, y)).mean()
        elif loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'ce':
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented.")

    def forward(self, 
                x: any, 
                num_neighbors: int=5, 
                remain_signal_ratio: float=0.1, 
                memory_mode: str='random', 
                return_all: bool = False) -> dict:
        """
        Args:
            x: Tensor of shape [B, 3, W, H], input images.
            num_neighbors: number of nearest neighbors to retrieve.
            remain_signal_ratio: float, ratio of original signal to retain.
        Returns:
            loss: float, loss between cls_signal and cls_memory.
            cls_memory: cls embedding of shape [B, 1, D].
        """
        cls_signal, nonnon_context_scores, non_context_embeddings = self.signal_encoder.SeperateContext(x)  # [B, 1, D],  [B, M, D]
        # update memory bank
        self.memory_bank.memorize(non_context_embeddings, nonnon_context_scores, mode=memory_mode)
        # Encode with memory encoder
        memory_embeddings = self.memory_encoder(non_context_embeddings, self.memory_bank, num_neighbors, remain_signal_ratio)  # [B, M*k, D] or [B, M*k+1, D]
        # calculate loss
        cls_memory = memory_embeddings[:, 0]  # [B, D]
        loss = self.loss_fn(cls_signal, cls_memory)
        if return_all:
            return {'embeddings': memory_embeddings, 'loss': loss}
        return {'cls_embeddings': cls_memory, 'loss': loss}