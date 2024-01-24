import torch
import pytest

from vllm.model_executor.utils import set_random_seed
from vllm.model_executor.layers.sampler import _cal_probs_sum

DTYPE = [torch.float16, torch.float32]
RANDOM_SEEDS = list(range(256))
VOCAB_SIZE = [160, 320, 640, 1280]
BATCH_SIZE = [1, 2, 4, 8, 32, 64, 128]
MATMUL_SIZE = [16, 32, 64, 128, 256]


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("matmul_size", MATMUL_SIZE)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cumsum(seed: int, batch_size: int, vocab_size: int, matmul_size: int,
                dtype: torch.dtype):
    set_random_seed(seed)
    fake_logits = torch.randn(batch_size,
                              vocab_size,
                              device="cuda",
                              dtype=dtype)
    probs = torch.softmax(fake_logits, dim=-1)
    probs, _ = probs.sort(dim=-1, descending=True)
    probs1 = torch.cumsum(probs, dim=-1)
    probs2 = _cal_probs_sum(probs, matmul_size=matmul_size)
    assert torch.allclose(probs1, probs2, atol=1e-2)
