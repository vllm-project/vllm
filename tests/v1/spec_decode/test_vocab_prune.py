import torch
import pytest
from unittest.mock import patch

from vllm.v1.spec_decode.eagle import load_vocab_freq, prune_draft_vocab

@pytest.mark.parametrize("mock_tensor", [
    torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32),
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32),
    torch.tensor([100, 50, 25, 12, 6, 3], dtype=torch.float32)
])
@patch("vllm.v1.spec_decode.eagle.hf_hub_download")
@patch("vllm.v1.spec_decode.eagle.torch.load")
def test_load_vocab_freq(mock_torch_load, mock_hf_hub_download, mock_tensor):
    # Mock download & load
    mock_hf_hub_download.return_value = "mock_draft_vocab_freq.pt"
    mock_torch_load.return_value = mock_tensor

    vocab_freq = load_vocab_freq("user/repo/mock_file.pt")

    # Ensure tensor is loaded and converted correctly
    assert isinstance(vocab_freq, torch.Tensor)
    assert vocab_freq.dtype == torch.int64
    assert vocab_freq.numel() == mock_tensor.numel()
    assert torch.all(vocab_freq >= 0)


def test_prune_draft_vocab():
    # Frequencies designed to test cumulative mass
    vocab_freqs = torch.ones(10, dtype=torch.float32)
    vocab_freqs[9] = 9.0  # 50% of total mass

    # 50% threshold: only the largest token
    pruned_vocab = prune_draft_vocab(vocab_freqs, 0.5)
    assert torch.equal(pruned_vocab, torch.tensor([9]))

    # Slightly above 50%: include one more
    pruned_vocab = prune_draft_vocab(vocab_freqs, 0.51)
    assert torch.equal(pruned_vocab, torch.tensor([9, 0]))

    # Near total mass: include all tokens
    pruned_vocab = prune_draft_vocab(vocab_freqs, 0.99)
    expected = torch.tensor([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert torch.equal(pruned_vocab, expected)
