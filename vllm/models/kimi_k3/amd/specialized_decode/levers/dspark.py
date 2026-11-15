# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers on vLLM's DFlash-family speculators.

``VLLM_K31_BEAT_MARKOV``: ``DSparkSpeculator._sample_sequential`` runs the
Markov steps of a Kimi-K3 DSpark draft as one fused launch each
(``ops/markov_sample``). Not through the draft model's ``sample_sequential``
hook: that would turn off the batch-sharded draft logits cache, which these
launches honour.

``VLLM_K31_BEAT_DRAFTMETA``: ``DFlashSpeculator.set_attn`` hands the model state
the drafter's layers, so the target step skips the attention groups that hold
only them (``KimiK3ModelState.target_attn_groups``)."""

import functools

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.levers import MARKOV, lever_on, wrap

logger = init_logger(__name__)


def install() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    wrap(DSparkSpeculator, "_sample_sequential", _wrap_sample_sequential)
    wrap(DFlashSpeculator, "set_attn", _wrap_set_attn)


def fused_markov_ok(spec, num_reqs: int) -> bool:
    """Whether the fused Markov launches take this draft step's loop."""
    from vllm.platforms import current_platform

    if not (lever_on(MARKOV) and current_platform.is_rocm()):
        return False
    if spec._draft_topk is not None or hasattr(spec.model, "sample_sequential"):
        return False
    from vllm.models.kimi_k3.amd.ops import markov_sample

    why = markov_sample.why_not(spec, num_reqs)
    if why is not None:
        logger.info_once("DSpark fused Markov sampling off: %s", why)
        return False
    return True


def sample_sequential(spec, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """``DSparkSpeculator._sample_sequential`` when ``fused_markov_ok``: the
    base logits, then the fused launches (or, if they decline the base logits,
    vLLM's loop without the confidence head and logit transform ``why_not``
    excludes)."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    if markov_sample.run(spec, num_reqs, base_logits):
        return
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        bias = spec.model.markov_bias(spec.model.markov_embed(prev))
        draft_sampled_i = spec._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i


def _wrap_sample_sequential(stock):
    @functools.wraps(stock)
    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if fused_markov_ok(self, num_reqs):
            sample_sequential(self, num_reqs, head_hidden)
        else:
            stock(self, num_reqs, head_hidden)

    return _sample_sequential


def _wrap_set_attn(stock):
    @functools.wraps(stock)
    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ) -> None:
        stock(
            self,
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        note = getattr(model_state, "note_draft_attn_layers", None)
        if note is not None:
            note(self.draft_attn_layer_names)

    return set_attn
