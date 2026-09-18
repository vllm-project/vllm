# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the DeepSeek V4 MTP drafter.

The speculator opts into ``use_local_argmax_reduction`` by probing the draft
model for ``get_top_tokens``, and expects it to select the same greedy draft
token the full-vocab argmax would. V4 reaches its LM head through hc_head and
the shared head's RMSNorm, so both methods have to share that transform.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.models.deepseek_v4.amd.mtp import (
        DeepSeekV4MTP,
        DeepSeekV4MultiTokenPredictor,
    )
elif current_platform.is_xpu():
    from vllm.models.deepseek_v4.xpu.mtp import (
        DeepSeekV4MTP,
        DeepSeekV4MultiTokenPredictor,
    )
else:
    from vllm.models.deepseek_v4.nvidia.mtp import (
        DeepSeekV4MTP,
        DeepSeekV4MultiTokenPredictor,
    )

DEVICE_TYPE = current_platform.device_type
START_LAYER_IDX = 61
HIDDEN_SIZE = 4096
HC_MULT = 4
RMS_NORM_EPS = 1e-6
HC_EPS = 1e-6
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="hc_head and the shared-head RMSNorm are device kernels",
)


def _make_predictor(vocab_size: int) -> DeepSeekV4MultiTokenPredictor:
    config = PretrainedConfig()
    config.hidden_size = HIDDEN_SIZE
    config.vocab_size = vocab_size
    config.rms_norm_eps = RMS_NORM_EPS

    # A single vocab shard, so the reduction returns the local argmax without
    # touching the process group.
    with (
        mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        shared_head = SharedHead(config=config, prefix="shared_head")

    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        shared_head.head.weight.normal_(0.0, 0.02, generator=generator)
        # The default norm weight is all ones, and the argmax is invariant to
        # the norm's per-row rescale, so a uniform gain would hide a skipped
        # shared_head. Vary it so the comparison below can see one.
        shared_head.norm.weight.normal_(1.0, 0.5, generator=generator)
    shared_head = shared_head.to(device=DEVICE_TYPE, dtype=torch.bfloat16)

    hc_head_fn = torch.randn(
        HC_MULT, HC_MULT * HIDDEN_SIZE, dtype=torch.float32, generator=generator
    )
    # Non-trivial gate scale and bias: hc_head mixes the hc_mult streams with
    # these, and defaults of 0 would make a skipped hc_head harder to see.
    hc_head_scale = torch.randn(1, dtype=torch.float32, generator=generator) * 0.1
    hc_head_base = torch.randn(HC_MULT, dtype=torch.float32, generator=generator) * 0.1

    mtp_layer = SimpleNamespace(
        config=config,
        hc_mult=HC_MULT,
        hc_eps=HC_EPS,
        rms_norm_eps=RMS_NORM_EPS,
        hc_head_op=HCHeadOp(),
        hc_head_fn=(hc_head_fn * 1e-4).to(DEVICE_TYPE),
        hc_head_scale=hc_head_scale.to(DEVICE_TYPE),
        hc_head_base=hc_head_base.to(DEVICE_TYPE),
        shared_head=shared_head,
    )

    # Only the attributes the two methods read; a real predictor would also
    # build the MLA/MoE draft block, which is irrelevant here.
    predictor = object.__new__(DeepSeekV4MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.layers = {str(START_LAYER_IDX): mtp_layer}
    predictor.mtp_start_layer_idx = START_LAYER_IDX
    predictor.num_mtp_layers = 1
    predictor.logits_processor = LogitsProcessor(vocab_size)
    return predictor


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    """The flat pre-hc_head residual the MTP layer returns."""
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HC_MULT * HIDDEN_SIZE, dtype=torch.float32, generator=generator
    )
    return hidden_states.to(device=DEVICE_TYPE, dtype=torch.bfloat16)


def _assert_selects_max_logit(top: torch.Tensor, logits: torch.Tensor) -> None:
    """The selected token must carry the row's maximum logit.

    hc_head emits bf16, so rows can tie for the maximum and then no single
    index is the right answer; compare the values the two paths select.
    """
    assert top.shape == (logits.shape[0],)
    selected = logits.gather(1, top.unsqueeze(1)).squeeze(1)
    assert torch.equal(selected, logits.max(dim=-1).values)


def test_get_top_tokens_matches_full_argmax(default_vllm_config):
    """Same draft token as the path taken when the reduction is off.

    ``get_top_tokens`` has to run the same pre-head transform as
    ``compute_logits``: hc_head over the (T, hc_mult, D) residual, then the
    shared head's RMSNorm. Dropping either one returns a different draft token
    instead of failing, which is what this comparison catches.
    """
    predictor = _make_predictor(vocab_size=512)

    for num_tokens in (1, 16, 128):
        hidden_states = _hidden_states(num_tokens, seed=11)

        logits = predictor.compute_logits(hidden_states)
        top = predictor.get_top_tokens(hidden_states)

        assert top.dtype == torch.int64
        _assert_selects_max_logit(top, logits)


def test_get_top_tokens_honors_padded_vocab(default_vllm_config):
    """Padding entries above ``org_vocab_size`` must never win the argmax."""
    vocab_size = 500
    predictor = _make_predictor(vocab_size)
    head = predictor.layers[str(START_LAYER_IDX)].shared_head.head
    assert head.shard_indices.num_org_vocab_padding > 0

    hidden_states = _hidden_states(32, seed=13)
    logits = predictor.compute_logits(hidden_states)
    top = predictor.get_top_tokens(hidden_states)

    _assert_selects_max_logit(top, logits)
    assert (top < vocab_size).all()


def test_wrapper_forwards_spec_step_idx():
    """``DeepSeekV4MTP`` is what the speculator probes, so it needs the method."""
    assert hasattr(DeepSeekV4MTP, "get_top_tokens")

    hidden_states = torch.empty(0)
    wrapper = object.__new__(DeepSeekV4MTP)
    wrapper.model = SimpleNamespace(get_top_tokens=lambda *args: args)

    assert DeepSeekV4MTP.get_top_tokens(wrapper, hidden_states, 2) == (hidden_states, 2)
    assert DeepSeekV4MTP.get_top_tokens(wrapper, hidden_states) == (hidden_states, 0)
