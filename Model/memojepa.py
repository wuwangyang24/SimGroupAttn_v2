from Encoder.memory_encoder import MemoryEncoder
from Encoder.signal_encoder import SignalEncoder
from Memory.memory_bank import MemoryBank
import torch


class MemoryJepaEncoder:
    """ Memory JEPA Encoder combining MemoryEncoder and SignalEncoder."""
    def __init__(
        self,
        cfg: dict,
    ):
        signal_encoder_cfg = cfg.get("signal_encoder", {})
        memory_encoder_cfg = cfg.get("memory_encoder", {})
        self.signal_encoder = SignalEncoder(**signal_encoder_cfg)
        self.memory_encoder = MemoryEncoder(**memory_encoder_cfg)
        self.memory_bank = MemoryBank(
            capacity=cfg.get("memory_capacity", 10000),
            embed_dim=memory_encoder_cfg.get("embed_dim", 768),
            device=cfg.get("memory_device", 'gpu'),
            dtype=cfg.get("memory_dtype", torch.float16)
        )
        self.loss_fn = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x: any, k: int, remain_signal_ratio: float=0.1) -> dict:
        """ Forward function.
        Args:
            x: Tensor of shape [B, M, D], input images.
            k: number of nearest neighbors to retrieve.
            remain_signal_ratio: float, ratio of original signal to retain.
        Returns:
            loss: float, loss between cls_signal and cls_memory.
            cls_memory: cls embedding of shape [B, 1, D].
        """
        cls_signal, non_context_embeddings = self.signal_encoder.SeperateContext(x)  # [B, 1, D],  [B, M, D]
        # If memory bank is empty, skip memory retrieval
        if self.memory_bank.stored_size == 0:
            k = 0 # no memory retrieval
            remain_signal_ratio = 1.0 # keep all signal
        # Encode with memory encoder
        y = self.memory_encoder(non_context_embeddings, self.memorybank, k, remain_signal_ratio)  # [B, M*k, D] or [B, M*k+1, D]
        cls_memory = y[:, 0]  # [B, D]
        loss = self.loss_fn(cls_signal, cls_memory)
        # update memory bank
        self.memory_bank.memorize(non_context_embeddings, mode="random")
        return {'cls': cls_memory, 'loss': loss}