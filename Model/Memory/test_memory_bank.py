import pytest
import torch
from unittest.mock import MagicMock, patch
from recollect_faiss import RecollectFaiss
from memory_bank import MemoryBank  # adjust path if needed


@pytest.fixture
def mock_recollector():
    """Mock RecollectFaiss to avoid using actual FAISS during tests."""
    mock = MagicMock(spec=RecollectFaiss)
    mock.recollect.return_value = (
        torch.zeros((4,)),  # distances
        torch.zeros((4,), dtype=torch.long),  # indices
    )
    return mock


@pytest.fixture
def memory_bank(mock_recollector):
    """Create a small memory bank for testing."""
    with patch("Model.Memory.memory_bank.RecollectFaiss", return_value=mock_recollector):
        mb = MemoryBank(capacity=10, embed_dim=4, device="cpu", dtype=torch.float32)
    return mb


def test_initialization(memory_bank):
    """Test that memory bank initializes correctly."""
    assert memory_bank.memory.shape == (10, 4)
    assert memory_bank.scores.shape == (10,)
    assert memory_bank.stored_size == 0
    assert isinstance(memory_bank.recollector, RecollectFaiss)


def test_memorize_fills_memory(memory_bank):
    """Test that memory is filled correctly until capacity."""
    items = torch.randn(5, 4)
    scores = torch.randn(5)
    memory_bank.memorize(items, scores)
    assert memory_bank.stored_size == 5
    torch.testing.assert_close(memory_bank.memory[:5], items)
    torch.testing.assert_close(memory_bank.scores[:5], scores)


def test_memorize_handles_overflow_random(memory_bank):
    """Test that memory overflow replaces random elements."""
    items = torch.randn(12, 4)
    scores = torch.randn(12)
    memory_bank.memorize(items, scores, mode="random")
    assert memory_bank.stored_size == memory_bank.capacity  # should be full


def test_memorize_replow_replaces_lowest(memory_bank):
    """Test that 'replow' replaces lowest scoring entries."""
    # Fill memory
    memory_bank.memorize(torch.randn(10, 4), torch.arange(10.0))
    # New items with high scores should replace low ones
    new_items = torch.randn(3, 4)
    new_scores = torch.ones(3) * 100.0
    memory_bank.memorize(new_items, new_scores, mode="replow")
    # Check that at least some of the lowest scores were replaced
    assert (memory_bank.scores > 10.0).any()


def test_recollect_returns_expected_shapes(memory_bank, mock_recollector):
    """Test that recollect returns distances and embeddings of correct shapes."""
    B, M, D, K = 2, 2, 4, 2
    # Fill memory first
    items = torch.randn(10, 4)
    scores = torch.randn(10)
    memory_bank.memorize(items, scores)

    query = torch.randn(B, M, D)
    distances, embeddings = memory_bank.recollect(query, k=K)

    assert distances.shape == (B, M * K)
    assert embeddings.shape == (B, M * K, D)
    mock_recollector.update_index.assert_called_once()


def test_recollect_raises_if_empty(memory_bank):
    """Ensure recollect() raises if memory is empty."""
    query = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="Memory bank is empty"):
        memory_bank.recollect(query, k=1)


def test_reset_clears_memory(memory_bank):
    """Ensure reset() clears stored_size and reinitializes tensors."""
    memory_bank.memorize(torch.randn(5, 4), torch.randn(5))
    memory_bank.reset()
    assert memory_bank.stored_size == 0
    assert memory_bank.memory.shape == (10, 4)
    assert memory_bank.scores.shape == (10,)


def test_get_memory_returns_valid_part(memory_bank):
    """Ensure get_memory returns only the filled part."""
    items = torch.randn(5, 4)
    scores = torch.randn(5)
    memory_bank.memorize(items, scores)
    mem = memory_bank.get_memory()
    assert mem.shape == (5, 4)
    torch.testing.assert_close(mem, items)
