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

    def update_index(self, memory: torch.Tensor) -> None:
        """Update the FAISS index with new memory embeddings.
         Args:
            memory (torch.Tensor): Memory embeddings of shape (M, D).
        """
        self.index.reset()
        self.index.add(memory)

    def recollect(self, query: torch.Tensor, k: int) -> torch.Tensor:
        """Retrieve top-k nearest neighbors for the given query embeddings.
         Args:
            query (torch.Tensor): Query embeddings of shape (N, D).
            k (int): Number of nearest neighbors to retrieve.
        Returns:
            torch.Tensor: Distances and indices of the top-k nearest neighbors. (N, k)
        """
        D, I = self.index.search(query, k)
        return D, I