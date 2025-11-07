import torch
import faiss
import faiss.contrib.torch_utils

class MemorySearcherFaiss:
    def __init__(self, embed_dim: int, device='gpu') -> None:
        self.device = device
        # GPU resources
        res = faiss.StandardGpuResources() 
        # Inner product index 
        self.index = faiss.IndexFlatIP(embed_dim)
        if device.startswith('cuda') or device == 'gpu':
            # Move index to GPU 0
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def update_index(self, memory: torch.Tensor) -> None:
        self.index.reset()
        self.index.add(memory.to(self.device))

    def search(self, query: torch.Tensor, k: int) -> torch.Tensor:
        D, I = self.index.search(query, k)
        device = query.device
        return torch.from_numpy(D).to(self.device), torch.from_numpy(I).to(self.device)