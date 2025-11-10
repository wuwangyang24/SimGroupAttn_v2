import torch

def SeperateContext(attn_scores: torch.Tensor, context_ratio: float) -> torch.Tensor:
    """Seperate attention scores into context and non-context parts.
    Args:
        attn_scores: Tensor of shape [B, H, N, N], attention scores.
        context_ratio: float, ratio of context patches to total patches.
    Returns:
        context_mask: Tensor of shape [B, M], binary mask indicating context patches.
        non_context_mask: Tensor of shape [B, N-M], binary mask indicating non-context patches.
    """
    B, N, _ = attn_scores.shape
    # Compute the mean attention score for each node
    mean_scores = attn_scores.mean(dim=-1)  # [B, N]
    # Determine a threshold to separate context and non-context nodes
    threshold = mean_scores.mean(dim=-1, keepdim=True)  # [B, 1]
    # Create binary masks based on the threshold
    context_mask = (mean_scores >= threshold).float()  # [B, N]
    object_mask = (mean_scores < threshold).float()   # [B, N]
    return context_mask, object_mask
    