# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-model side of the non-layer amd_kernels routes: set-up at the end of weight
loading (collective, same order on every TP rank) and the `compute_logits` hooks."""

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import (
    akr_embed,
    akr_final,
    akr_lmhead,
    akr_nonlayer,
    akr_sampling,
)

logger = init_logger(__name__)


def setup(model) -> None:
    """From `KimiLinearForCausalLM.process_weights_after_loading`."""
    if get_pp_group().world_size != 1:
        return
    with set_current_vllm_config(model.vllm_config):
        akr_embed.install(model.model.embed_tokens)
        model._akr_lmhead = akr_lmhead.setup(
            "target", model.lm_head, model.logits_processor, model.vllm_config
        )
        akr_sampling.install_top_p(model.lm_head.org_vocab_size)
        akr_final.setup(model.model)


def compute_logits(model, normed: torch.Tensor) -> torch.Tensor | None:
    """Gathered logits of the final-normed rows on the kernel path, else None."""
    if not getattr(model, "_akr_lmhead", False):
        return None
    return akr_lmhead.logits("target", model.lm_head, normed)


def compute_logits_owned(model, hidden_states, sorted_logits_indices, metadata):
    """Model-runner hook (batch-sharded sampling, `VLLM_KIMI_AKR_SHARDED=1`): the
    logits rows this rank samples, or None for the stock all-to-all."""
    if not getattr(model, "_akr_lmhead", False) or not akr_nonlayer.switch("SHARDED"):
        return None
    return akr_lmhead.owned_logits(
        model.lm_head, model.model.norm, hidden_states, sorted_logits_indices, metadata
    )
