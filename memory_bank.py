import torch

class MemoryBank:
    def __init__(self, capacity: int, embed_dim: int, device='cpu', dtype=torch.float32) -> None:
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype
        self.memory = torch.zeros(capacity, embed_dim, device=device, dtype=dtype)
        self.stored_size = 0

    def add(self, items: torch.Tensor) -> None:
        """Add new embeddings to the memory bank."""
        items = items.to(self.device, self.dtype)
        n = items.shape[0]
        # space left in bank
        space_left = max(self.capacity - self.stored_size, 0)
        # fill available space first
        fill = min(space_left, n)
        if fill > 0:
            self.memory[self.stored_size:self.stored_size+fill] = items[:fill]
            self.stored_size += fill
        # if overflow, randomly replace existing items
        overflow = n - fill
        if overflow > 0:
            idx = torch.randint(0, self.capacity, (overflow,), device=self.device)
            self.memory[idx] = items[fill:]

    def clear(self) -> None:
        """Reset memory bank in-place."""
        self.memory.zero_()
        self.stored_size = 0

    def get_memory(self) -> torch.Tensor:
        """Return the valid part of the memory bank."""
        return self.memory[:self.stored_size] if self.stored_size < self.capacity else self.memory
