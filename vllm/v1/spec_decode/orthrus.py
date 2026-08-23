# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXPERIMENTAL Orthrus diffusion-mode proposer for speculative decoding.

Status: unvalidated milestone-3 attempt (see the discussion on
https://github.com/vllm-project/vllm/pull/44792). This is NOT wired up
end-to-end and has not been run against vLLM's real scheduler/CI.

What this implements: loading a second copy of the target Orthrus
checkpoint as a speculative-decode "draft" model, and KV-sharing each of
its diffusion attention layers (``self_attn.attn_diff``, added in
``vllm/model_executor/models/orthrus.py``) with the corresponding
same-index *target* model layer's ``self_attn.attn`` -- so a diffusion
forward reads the target's real, already-populated paged KV cache instead
of the offline tensors used for the milestone-1/2 validation in
``OrthrusForCausalLM.forward_diffusion`` / ``generate_with_diffusion``.
This part mirrors ``Gemma4Proposer._setup_gemma4_kv_sharing``, an
already-validated pattern, just with simpler 1:1 same-index layer mapping
(Orthrus's diffusion layers are literally paired with same-architecture
target layers, unlike Gemma4's cross-architecture MTP head).

What is deliberately NOT implemented here: ``propose()``'s
input/attention-metadata construction for the one-shot "whole block in a
single forward" proposal that gives Orthrus its actual speedup (see
DFlashProposer.set_inputs_first_pass for the closest existing analogue,
which relies on a custom Triton kernel and deep knowledge of
CommonAttentionMetadata's slot-mapping/cu-seqlens semantics that could
not be safely guessed at without live iteration against a running
engine -- getting this wrong risks writing into the wrong paged-cache
slots under real multi-request batching, not just producing wrong
output). Left as an explicit, scoped follow-up.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class OrthrusProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "orthrus"
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )

    @override
    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        self._setup_orthrus_kv_sharing(target_model)

    def _setup_orthrus_kv_sharing(self, target_model: torch.nn.Module) -> None:
        """Map each draft layer's diffusion attention to its same-index
        target layer's AR attention, so a diffusion forward reads the
        target's real, already-populated paged KV cache."""
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return
        if not (hasattr(target_model, "model") and hasattr(target_model.model, "layers")):
            return

        target_prefix = None
        for name, _ in target_model.named_modules():
            if name.endswith(".self_attn.attn"):
                target_prefix = name.rsplit(".layers.", 1)[0] + ".layers"
                break
        if target_prefix is None:
            logger.warning(
                "OrthrusProposer: could not find target attention layer "
                "names for KV sharing; diffusion pass will not see the "
                "target's real KV cache."
            )
            return

        for idx, layer in enumerate(self.model.model.layers):
            attn_diff = getattr(layer.self_attn, "attn_diff", None)
            if attn_diff is None:
                continue
            target_layer_name = f"{target_prefix}.{idx}.self_attn.attn"
            attn_diff.kv_sharing_target_layer_name = target_layer_name
            logger.info(
                "OrthrusProposer: draft layer %d attn_diff -> %s",
                idx,
                target_layer_name,
            )

    @override
    def propose(self, *args, **kwargs):
        raise NotImplementedError(
            "OrthrusProposer.propose() is not yet implemented. load_model's "
            "KV-sharing wiring is in place, but the one-shot block-proposal "
            "input/attention-metadata construction (see this module's "
            "docstring) needs to be built and validated against a running "
            "engine before this proposer can actually run."
        )
