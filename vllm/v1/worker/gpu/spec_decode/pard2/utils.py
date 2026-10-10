# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import ParallelConfig, VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


def _get_pard2_parallel_config(
    parallel_config: ParallelConfig,
    tensor_parallel_size: int,
) -> ParallelConfig:
    if parallel_config.enable_eplb:
        logger.warning_once(
            "EPLB is disabled for the PARD-2 draft model. EPLB remains enabled "
            "for the target model."
        )
    return replace(
        parallel_config,
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        enable_eplb=False,
        eplb_config=replace(
            parallel_config.eplb_config,
            num_redundant_experts=0,
        ),
        enable_elastic_ep=False,
    )


def load_pard2_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    """Load a PARD-2 draft as a standalone LM.

    The draft ships its own embeddings and lm_head, so the sharing helpers below
    leave them alone; they run only to cover checkpoints that omit one.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
        maybe_share_target_embed,
    )

    draft_vllm_config = replace(
        speculative_config.apply_draft_overrides(vllm_config),
        parallel_config=_get_pard2_parallel_config(
            vllm_config.parallel_config,
            speculative_config.draft_parallel_config.tensor_parallel_size,
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    # VllmConfig post-init restores the target's quant config, which the draft
    # config is kept around for (pard2_target_layers). Against a quantized target
    # that hands the bf16 draft INT8 scales no checkpoint loads, so acceptance
    # collapses to 1.0.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("pard2_draft"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
