# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers on vLLM's DFlash-family speculators.

``VLLM_K31_BEAT_MARKOV``: ``DSparkSpeculator._sample_sequential`` runs the
Markov steps of a Kimi-K3.1 DSpark draft (the preview's MLA draft or the
release's GQA sliding-window draft, ``markov_sample.markov_drafts``) as one fused
launch each (``ops/markov_sample``), greedy or probabilistic drafts, with the
confidence head when the speculator uses it. Not through the draft model's
``sample_sequential`` hook: that would turn off the batch-sharded draft logits
cache, which these launches honour.

``VLLM_K31_BEAT_DRAFTMETA``: ``DFlashSpeculator.set_attn`` hands the model state
the drafter's layers, so the target step skips the attention groups that hold
only them (``KimiK3ModelState.target_attn_groups``), and logs the drafter's
attention: an MLA draft's AITER MLA groups take the bs1 planner
(``levers.mla``), a GQA draft's groups (KIMI_K3_FLYDSL_SWA_GQA or
TRITON_ATTN on ROCm) plan no persistent schedule, so there is nothing for it
to replace."""

import functools

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.levers import (
    DRAFTMETA,
    MARKOV,
    lever_on,
    wrap,
)

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
    vLLM's loop without the logit transform ``why_not`` excludes), then the
    confidence head as vLLM's loop computes it."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    from vllm.models.kimi_k3.amd.specialized_decode.levers import tpgather

    tp = tpgather.markov_tp(num_reqs, base_logits.shape[-1])
    if markov_sample.run(spec, num_reqs, base_logits, tp):
        embeds = None
    else:
        idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
        prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
        embeds = []
        for i in range(n_spec):
            embed = spec.model.markov_embed(prev)
            embeds.append(embed)
            draft_sampled_i = spec._sample_logits(
                base_logits[:, i] + spec.model.markov_bias(embed),
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            spec.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i
    if spec.use_confidence_head:
        if embeds is None:
            # Step i's Markov embedding is of its previous token: the anchor,
            # then the drafted tokens (an exact lookup, as in vLLM's loop).
            anchor = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
            drafted = spec.draft_tokens[:num_reqs, : n_spec - 1]
            prev = torch.cat((anchor[:, None].to(drafted.dtype), drafted), dim=1)
            embed = spec.model.markov_embed(prev)
        else:
            embed = torch.stack(embeds, dim=1)
        confidence = spec.model.compute_confidence(sample_hidden, embed.flatten(0, 1))
        spec.draft_token_confidence_probs[:num_reqs] = confidence.view(
            num_reqs, n_spec
        )


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
            log_draft_attn(self)

    return set_attn


def draft_attn_backends(spec) -> dict[str, bool]:
    """The drafter's attention backends, each with whether it is MLA."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    out: dict[str, bool] = {}
    for groups in getattr(spec, "attn_groups", ()):
        for group in groups:
            if not set(group.layer_names) & set(spec.draft_attn_layer_names):
                continue
            name = group.backend.get_name()
            out[name] = out.get(name, False) or isinstance(
                group.kv_cache_spec, MLAAttentionSpec
            )
    return out


def log_draft_attn(spec) -> None:
    """``VLLM_K31_BEAT_DRAFTMETA``'s view of the drafter, once a process."""
    if not lever_on(DRAFTMETA):
        return
    backends = draft_attn_backends(spec)
    names = ", ".join(sorted(backends)) or "none"
    if any(backends.values()):
        planner = "its AITER MLA decode metadata takes the bs1 MLA planner"
    else:
        planner = (
            "a non-MLA draft: no AITER MLA schedule to plan, the bs1 MLA "
            "planner is not used"
        )
    logger.info_once(
        "DSpark draft attention (%s, %d layers): the target step skips its "
        "draft-only groups; %s",
        names, len(spec.draft_attn_layer_names), planner,
    )  # fmt: skip
