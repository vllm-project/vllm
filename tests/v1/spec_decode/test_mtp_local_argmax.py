# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the generic MTP drafter (``deepseek_mtp.py``).

The speculator opts into ``use_local_argmax_reduction`` by probing the draft
model for ``get_top_tokens``, and expects it to select the same greedy draft
token the full-vocab argmax would.
"""

from types import SimpleNamespace
from unittest import mock

import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    SharedHead,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type
START_LAYER_IDX = 61
HIDDEN_SIZE = 64
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def _make_predictor(vocab_size: int) -> DeepSeekMultiTokenPredictor:
    config = PretrainedConfig()
    config.hidden_size = HIDDEN_SIZE
    config.vocab_size = vocab_size
    config.rms_norm_eps = 1e-6

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
    shared_head = shared_head.to(device=DEVICE_TYPE, dtype=torch.float32)

    # Only the four attributes the two methods read; a real predictor would
    # also build the MLA/MoE draft block, which is irrelevant here.
    predictor = object.__new__(DeepSeekMultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.layers = {str(START_LAYER_IDX): SimpleNamespace(shared_head=shared_head)}
    predictor.mtp_start_layer_idx = START_LAYER_IDX
    predictor.num_mtp_layers = 1
    predictor.logits_processor = LogitsProcessor(vocab_size)
    return predictor


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, HIDDEN_SIZE, generator=generator).to(DEVICE_TYPE)


def test_get_top_tokens_matches_full_argmax(default_vllm_config):
    """Same draft token as the path taken when the reduction is off.

    ``get_top_tokens`` has to apply ``shared_head`` (the final RMSNorm) just as
    ``compute_logits`` does. The v3.2 MTP hands its LM head an already-normed
    hidden state and so legitimately omits this; reusing that form here would
    silently change the draft token rather than fail, which is what this
    comparison catches.
    """
    predictor = _make_predictor(vocab_size=512)

    for num_tokens in (1, 16, 128):
        hidden_states = _hidden_states(num_tokens, seed=11)

        expected = predictor.compute_logits(hidden_states).argmax(dim=-1)
        top = predictor.get_top_tokens(hidden_states)

        assert top.dtype == torch.int64
        assert torch.equal(top, expected)


def test_get_top_tokens_honors_padded_vocab(default_vllm_config):
    """Padding entries above ``org_vocab_size`` must never win the argmax."""
    vocab_size = 500
    predictor = _make_predictor(vocab_size)
    head = predictor.layers[str(START_LAYER_IDX)].shared_head.head
    assert head.shard_indices.num_org_vocab_padding > 0

    hidden_states = _hidden_states(32, seed=13)
    top = predictor.get_top_tokens(hidden_states)

    assert torch.equal(top, predictor.compute_logits(hidden_states).argmax(dim=-1))
    assert (top < vocab_size).all()


def test_wrapper_forwards_spec_step_idx():
    """``DeepSeekMTP`` is what the speculator probes, so it needs the method."""
    assert hasattr(DeepSeekMTP, "get_top_tokens")

    hidden_states = torch.empty(0)
    wrapper = object.__new__(DeepSeekMTP)
    wrapper.model = SimpleNamespace(get_top_tokens=lambda *args: args)

    assert DeepSeekMTP.get_top_tokens(wrapper, hidden_states, 2) == (hidden_states, 2)
    assert DeepSeekMTP.get_top_tokens(wrapper, hidden_states) == (hidden_states, 0)
