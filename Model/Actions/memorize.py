import torch

def memorize(memory_bank: any, x_to_memory: torch.Tensor, score_to_memory: torch.Tensor, mem_eff: float, mode: str='random') -> None:
    """ Memorize selected patches into the memory bank.
    Args:
        memory_bank: memory bank object.
        x_to_memory (torch.Tensor): Tensor of shape (B, N, D) containing patches to consider for memorization.
        score_to_memory (torch.Tensor): Tensor of shape (B, N) containing scores for each patch.
        mem_eff (float): Memory efficiency ratio (0 < mem_eff <= 1).
        mode (str): Mode of selection ('random' or 'score').
    """
    B, N, _ = x_to_memory.shape
    K = int(N * mem_eff)
    if mode == 'random':
        # 1. Generate random scores for each patch
        rand_scores = torch.rand(B, N)  # shape (B, N)
        _, idx = torch.topk(rand_scores, K, dim=1)  # shape (B, K)
        # 2. Advanced indexing to select patches
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, K)  # shape (B, K)
        x_to_memory = x_to_memory[batch_idx, idx]  # (B, K, D)
        memory_bank.add(x_to_memory)
    elif mode == 'score':
        # 1. Take top-K indices along the patch dimension based on provided scores
        _, idx = torch.topk(score_to_memory, K, dim=1)  # shape (B, K)
        # 2. Advanced indexing to select patches
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, K)  # shape (B, K)
        x_to_memory = x_to_memory[batch_idx, idx]  # (B, K, D)
        memory_bank.add(x_to_memory)
    else:
        raise NotImplementedError(f'Memorize mode {mode} not implemented.')