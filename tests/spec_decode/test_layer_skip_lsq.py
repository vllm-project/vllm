import torch
import torch.nn as nn
from vllm.spec_decode.early_exit_model_runner import load_lsq_head


def test_lsq_loader_good_path(tmp_path):
    vocab, hidden = 128, 64
    w = torch.randn(vocab, hidden, dtype=torch.float32)
    torch.save(w.cpu(), tmp_path / "h6.pt")

    head = load_lsq_head(str(tmp_path), layer=6, device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(head, nn.Linear)
    assert head.bias is None
    assert tuple(head.weight.shape) == (vocab, hidden)
    assert head.weight.dtype == torch.float32
    assert head.weight.device.type == "cpu"
    assert head.weight.requires_grad is False


def test_lsq_loader_missing_file(tmp_path):
    head = load_lsq_head(str(tmp_path), layer=7, device=torch.device("cpu"), dtype=torch.float32)
    assert head is None


def test_lsq_loader_bad_tensor_shape(tmp_path):
    torch.save(torch.randn(10).cpu(), tmp_path / "h5.pt")
    head = load_lsq_head(str(tmp_path), layer=5, device=torch.device("cpu"), dtype=torch.float32)
    assert head is None


def test_lsq_loader_no_path_is_ok():
    assert load_lsq_head(None, layer=3, device=torch.device("cpu"), dtype=torch.float32) is None
    assert load_lsq_head("", layer=3, device=torch.device("cpu"), dtype=torch.float32) is None