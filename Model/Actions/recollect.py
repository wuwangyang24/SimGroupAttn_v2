import torch
import faiss
import faiss.contrib.torch_utils


class RecollectFaiss:
    """A FAISS-based searcher for efficient nearest neighbor search."""
    def __init__(self, embed_dim: int, device='gpu') -> None:
        self.device = device
        # Inner product index 
        self.index = faiss.IndexFlatIP(embed_dim)
        if device.startswith('cuda') or device == 'gpu':
            # GPU resources
            res = faiss.StandardGpuResources() 
            # Move index to GPU 0
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.temp_memory = None

    def update_index(self, memory: torch.Tensor) -> None:
        self.index.reset()
        self.index.add(memory.to(self.device))

    def recollect(self, query: torch.Tensor, k: int) -> torch.Tensor:
        D, I = self.index.search(query, k)
        return torch.from_numpy(D).to(self.device), torch.from_numpy(I).to(self.device)