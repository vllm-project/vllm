# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`get_top_tokens` on the generic MTP drafter (`deepseek_mtp.py`).

The speculator opts into `use_local_argmax_reduction` by probing the draft
model for `get_top_tokens`, and expects it to select the same greedy draft
token as `compute_logits(...).argmax(-1)`.

Tensor parallelism is emulated in one process: every rank holds its own vocab
shard of the LM head, and the TP all-gather returns the concatenation of what
each rank actually contributed.
"""

from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import PreTrainedConfig

from vllm.model_executor.layers import logits_processor
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMultiTokenPredictor,
    SharedHead,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type
DTYPE = torch.bfloat16
START_LAYER_IDX = 61
NUM_MTP_LAYERS = 2
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


@pytest.fixture(autouse=True)
def _seed():
    set_random_seed(0)


def _make_rank_predictors(
    tp_size: int, vocab_size: int, hidden_size: int
) -> list[DeepSeekMultiTokenPredictor]:
    config = PreTrainedConfig()
    config.hidden_size = hidden_size
    config.vocab_size = vocab_size
    config.rms_norm_eps = 1e-6

    # One checkpoint shared by every rank, with a different head per MTP layer.
    head_weights = [
        torch.randn(vocab_size, hidden_size, device=DEVICE_TYPE) * 0.02
        for _ in range(NUM_MTP_LAYERS)
    ]
    # The argmax is invariant to the norm's per-row rescale, so the default
    # all-ones weight would hide a skipped shared_head; vary the gain.
    norm_weights = [
        torch.randn(hidden_size, device=DEVICE_TYPE) * 0.5 + 1.0
        for _ in range(NUM_MTP_LAYERS)
    ]

    predictors = []
    for rank in range(tp_size):
        layers = {}
        for i in range(NUM_MTP_LAYERS):
            with (
                mock.patch(
                    f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=rank
                ),
                mock.patch(
                    f"{_EMBEDDING}.get_tensor_model_parallel_world_size",
                    return_value=tp_size,
                ),
                torch.device(DEVICE_TYPE),
            ):
                shared_head = SharedHead(config=config, prefix="shared_head")
            head = shared_head.head
            assert head is not None and head.tp_size == tp_size
            head.weight_loader(head.weight, head_weights[i])
            # The loader zero-fills vocab padding, which a random hidden state
            # rarely beats anyway. Make the padding rows win unless masked.
            num_padding = head.shard_indices.num_org_vocab_padding
            if num_padding:
                head.weight.data[-num_padding:] = 100 * head.weight.data[0]
            shared_head.norm.weight.data.copy_(norm_weights[i])
            layers[str(START_LAYER_IDX + i)] = SimpleNamespace(
                shared_head=shared_head.to(DTYPE)
            )

        # Only the attributes the two methods read; a real predictor would
        # also build the MLA/MoE draft block, which is irrelevant here.
        predictor = object.__new__(DeepSeekMultiTokenPredictor)
        torch.nn.Module.__init__(predictor)
        predictor.layers = layers
        predictor.mtp_start_layer_idx = START_LAYER_IDX
        predictor.num_mtp_layers = NUM_MTP_LAYERS
        predictor.logits_processor = LogitsProcessor(vocab_size)
        predictors.append(predictor)
    return predictors


def _run_on_every_rank(
    predictors: list[DeepSeekMultiTokenPredictor],
    call: Callable[[DeepSeekMultiTokenPredictor], torch.Tensor],
) -> list[torch.Tensor]:
    """Run `call` on every rank against an emulated TP all-gather."""
    contributions: list[torch.Tensor] = []

    def record(local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        contributions.append(local)
        return torch.cat([local] * len(predictors), dim=dim)

    gather_calls = iter(range(len(predictors)))

    def gather(local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert torch.equal(local, contributions[next(gather_calls)])
        return torch.cat(contributions, dim=dim)

    results = []
    for collective in (record, gather):
        with mock.patch.multiple(
            logits_processor,
            tensor_model_parallel_all_gather=collective,
            tensor_model_parallel_gather=collective,
        ):
            results = [call(predictor) for predictor in predictors]
    return results


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
# 500 leaves vocab padding on the last shard; 129280 is DeepSeek-V3's vocab.
@pytest.mark.parametrize("vocab_size", [500, 129280])
@pytest.mark.parametrize("hidden_size", [64, 1024])
@pytest.mark.parametrize("num_tokens", [1, 16, 1024])
@pytest.mark.parametrize("spec_step_idx", [0, 1])
def test_get_top_tokens_matches_full_argmax(
    default_vllm_config,
    tp_size: int,
    vocab_size: int,
    hidden_size: int,
    num_tokens: int,
    spec_step_idx: int,
):
    """Every rank selects the token the full-vocab argmax would.

    `get_top_tokens` has to apply `shared_head` (the final RMSNorm) just as
    `compute_logits` does. The v3.2 MTP hands its LM head an already-normed
    hidden state and so legitimately omits this; reusing that form here would
    silently change the draft token rather than fail.
    """
    predictors = _make_rank_predictors(tp_size, vocab_size, hidden_size)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=DEVICE_TYPE, dtype=DTYPE
    )

    logits = _run_on_every_rank(
        predictors, lambda p: p.compute_logits(hidden_states, spec_step_idx)
    )[0]
    assert logits.shape == (num_tokens, vocab_size)
    expected = logits.argmax(dim=-1)

    for top in _run_on_every_rank(
        predictors, lambda p: p.get_top_tokens(hidden_states, spec_step_idx)
    ):
        assert top.dtype == torch.int64
        assert torch.equal(top, expected)
