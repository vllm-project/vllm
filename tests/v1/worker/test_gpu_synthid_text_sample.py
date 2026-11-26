# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for the SynthID-Text kernel", allow_module_level=True)

from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.synthid_text import synthid_text_logits
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.synthid_text import synthid_text_sample

DEVICE = current_platform.device_type
KEY = 0x0123_4567_89AB_CDEF
NUM_ROWS = 8


def _batch(vocab_size: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(NUM_ROWS, vocab_size, device=DEVICE) * 4
    # Rows 2 and 5 are top-k filtered.
    for row in (2, 5):
        kth = logits[row].topk(20).values[-1]
        logits[row] = logits[row].masked_fill(logits[row] < kth, float("-inf"))
    temperatures = torch.ones(NUM_ROWS, device=DEVICE)
    temperatures[6] = 0
    return {
        "logits": logits.to(dtype),
        "contexts": torch.randint(0, vocab_size, (NUM_ROWS, 4), device=DEVICE),
        "skip_mask": torch.tensor(
            [False, True, False, False, True, False, True, False], device=DEVICE
        ),
        "expanded_idx_mapping": torch.tensor([3, 1, 7, 0, 4, 2, 6, 5], device=DEVICE),
        "temperatures": temperatures,
        "seeds": torch.arange(NUM_ROWS, device=DEVICE) + 1000,
        "positions": torch.arange(NUM_ROWS, device=DEVICE) + 100,
    }


def _sample(batch: dict[str, torch.Tensor], depth: int, use_fp64: bool = False):
    return synthid_text_sample(
        batch["logits"],
        batch["contexts"],
        KEY,
        depth,
        batch["skip_mask"],
        batch["expanded_idx_mapping"],
        batch["temperatures"],
        batch["seeds"],
        batch["positions"],
        use_fp64=use_fp64,
    )


@pytest.mark.parametrize("vocab_size", [1000, 151936])
@pytest.mark.parametrize("depth", [1, 30])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_matches_reference_tournament_and_gumbel_sample(
    vocab_size: int, depth: int, dtype: torch.dtype, use_fp64: bool
):
    batch = _batch(vocab_size, dtype)
    logits = batch["logits"]
    watermarked = synthid_text_logits(logits, batch["contexts"], PhiloxPRF(KEY), depth)
    expected = gumbel_sample(
        torch.where(batch["skip_mask"].unsqueeze(-1), logits.float(), watermarked),
        batch["expanded_idx_mapping"],
        batch["temperatures"],
        batch["seeds"],
        batch["positions"],
        apply_temperature=False,
        is_drafting=False,
        use_fp64=use_fp64,
    )

    assert torch.equal(_sample(batch, depth, use_fp64), expected)


def test_rows_sample_the_same_token_in_any_batch():
    batch = _batch(151936, torch.float32)
    full = _sample(batch, 30)

    for row in range(NUM_ROWS):
        alone = {name: value[row : row + 1] for name, value in batch.items()}
        alone["temperatures"] = batch["temperatures"]
        alone["seeds"] = batch["seeds"]
        assert _sample(alone, 30).item() == full[row].item()
