import torch

def memorize(memory_bank: any, x: torch.Tensor, mem_eff: float) -> None:
        """Pushing embeddings to memory.
        Args:
            x (torch.Tensor): Embeddings of shape (B, N, D).
        """
        B, N, _ = x.shape
        K = int(N * mem_eff)
        # 1. Generate random scores for each patch
        rand_scores = torch.rand(B, N)  # shape (B, N)
        # 2. Take top-K indices along the patch dimension
        _, idx = torch.topk(rand_scores, K, dim=1)  # shape (B, K)
        # 3. Advanced indexing to select patches
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, K)  # shape (B, K)
        x = x[batch_idx, idx]  # (B, K, D)
        memory_bank.add(x)