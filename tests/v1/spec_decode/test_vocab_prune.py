import torch
from unittest import mock

from vllm.v1.spec_decode.eagle import load_vocab_freq, prune_draft_vocab

@pytest.mark.parametrize("vocab_freq", [
    torch.tensor([10, 20, 30, 40, 50], dtype=torch.float64),
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),
    torch.tensor([100, 50, 25, 12, 6, 3])
])
@patch("my_module.hf_hub_download")
@patch("my_module.torch.load")
def test_load_vocab_freq(mock_torch_load, mock_hf_hub_download, vocab_freq):

    # Mock the file download (doesn't matter what the path is)
    mock_hf_hub_download.return_value = "mock_draft_vocab_freq.pt"
    mock_torch_load.return_value = vocab_freq

    # prune the vocab
    draft_vocab = load_vocab_freq("mock_draft_vocab_freq.pt", prune_ratio)

    assert isinstance(draft_vocab, torch.Tensor)
    assert draft_vocab.dtype == torch.int64
    assert draft_vocab.numel() > 0


def test_prune_ratio():
    # Use frequencies where we can predict cumulative mass easily
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 9] -> total=18
    # Token 9 has 9/18 = 50% mass, others have 1/18 each
    vocab_freqs = torch.ones(10)
    vocab_freqs[9] = 9.0

    # 0.5 threshold should include only token 9 (exactly 50% mass)
    ret = prune_draft_vocab(vocab_freqs, 0.5)
    expected = torch.tensor([9])
    assert torch.equal(ret, expected)

    # 0.6 threshold should include token 9 + one more
    ret = prune_draft_vocab(vocab_freqs, 0.51)
    expected = torch.tensor([9, 0])
    assert torch.equal(ret, expected)

    # 0.99 threshold should include all tokens
    ret = prune_draft_vocab(vocab_freqs, 0.99)
    expected = torch.tensor([9, 0, 1, 2, 3, 4, 5, 6, 7, 8]) # tokens in descending frequency order
    assert torch.equal(ret, expected)
