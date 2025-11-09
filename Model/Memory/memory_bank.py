import torch


class MemoryBank:
    """A fixed-size memory bank for storing embeddings."""
    def __init__(self, capacity: int, embed_dim: int, device='cpu', dtype=torch.float16) -> None:
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.memory, self.scores, self.stored_size = self.reset()

    @torch.no_grad()
    def add(self, items: torch.Tensor, scores: torch.Tensor, mode: str = "random") -> None:
        """
        Add new embeddings to the memory bank.
        Args:
            items: Tensor of shape [N, D]
            scores: Tensor of shape [N]
            mode: "random" (replace random items) or "replow" (replace lowest-score items)
        """
        assert mode in {"random", "replow"}, f"Invalid mode: {mode}"

        items = items.to(self.device, self.dtype, non_blocking=True)
        scores = scores.to(self.device, self.dtype, non_blocking=True)
        n = items.size(0)
        # fill available space first
        if self.stored_size < self.capacity:
            fill = min(self.capacity - self.stored_size, n)
            end = self.stored_size + fill
            self.memory[self.stored_size:end].copy_(items[:fill])
            self.scores[self.stored_size:end].copy_(scores[:fill])
            self.stored_size = end
            if fill == n:
                return
            items = items[fill:]
            scores = scores[fill:]
        # overflow handling
        overflow = items.size(0)
        if mode == "random":
            idx = torch.randint(0, self.capacity, (overflow,), device=self.device)
        else:  # "replow"
            _, idx = torch.topk(self.scores, overflow, largest=False)
        self.memory[idx].copy_(items)
        self.scores[idx].copy_(scores)

    def reset(self) -> None:
        """Reset memory bank (Preallocate contiguous memory).
         Returns:
             memory: Tensor of shape [capacity, embed_dim]
             scores: Tensor of shape [capacity]
             stored_size: int
         """
        self.memory = torch.empty((self.capacity, self.embed_dim), device=self.device, dtype=self.dtype)
        self.scores = torch.empty(self.capacity, device=self.device, dtype=self.dtype)
        self.stored_size = 0
        return self.memory, self.scores, self.stored_size

    def get_memory(self) -> torch.Tensor:
        """Return the valid part of the memory bank."""
        return self.memory[:self.stored_size]
