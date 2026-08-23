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

``set_inputs_first_pass`` below implements the one-shot "whole block in a
single forward" proposal construction, adapted from
``DFlashProposer.set_inputs_first_pass`` but WITHOUT its Triton kernel --
plain PyTorch indexing, and restricted to batch_size == 1 (asserted at
runtime; multi-request batching is refused loudly rather than silently
mishandled). This piece is the highest-remaining-risk part of this whole
effort: it computes physical KV-cache slot indices from the block table
using the standard `physical_block_id * block_size + offset` formula (the
same one vLLM's own `_COMPUTE_SLOT_MAPPING_KERNEL` implements more
generally, see vllm/v1/worker/block_table.py), but has NOT been validated
against a running engine due to persistent Modal-side infrastructure
failures during testing (see PR discussion) -- treat as an unverified,
best-effort attempt for review, not a confirmed-working implementation.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
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
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        """Build the one-shot diffusion-block proposal input.

        UNVALIDATED (see module docstring). Restricted to batch_size == 1:
        multi-request batching would need per-request block-table indexing
        this simplified version does not implement, so it asserts rather
        than silently producing wrong slot mappings.

        Builds a block of ``num_speculative_tokens + 1`` positions
        continuing directly after the target's current sequence: position 0
        holds the last accepted/sampled token (``next_token_ids``), and the
        rest are filled with ``mask_token_id`` -- mirroring the reference
        implementation's diffusion-block construction (see
        ``modeling_orthrus.py``'s ``generate`` method), except here the
        block's tokens get real slot-mapped positions in the *target's*
        paged cache (via kv_sharing) rather than the reference's ephemeral,
        discarded KV.
        """
        batch_size = cad.batch_size()
        assert batch_size == 1, (
            "OrthrusProposer.set_inputs_first_pass only supports batch_size "
            "== 1 in its current form -- multi-request slot-mapping "
            "indexing is not implemented. Refusing rather than silently "
            "computing wrong physical KV-cache slots."
        )

        mask_token_id = self.vllm_config.model_config.hf_config.mask_token_id
        num_tokens = self.num_speculative_tokens + 1

        seq_len = int(cad.seq_lens[0].item())
        block_ids = torch.full(
            (num_tokens,), mask_token_id, dtype=torch.int32, device=self.device
        )
        block_ids[0] = next_token_ids[0]
        self.input_ids[:num_tokens] = block_ids

        block_positions = torch.arange(
            seq_len, seq_len + num_tokens, device=self.device, dtype=torch.int64
        )
        self._set_positions(num_tokens, block_positions)

        # Physical slot for each new position: standard paged-attention
        # convention (block_table[pos // block_size] * block_size + pos %
        # block_size) -- see vllm/v1/worker/block_table.py's
        # _COMPUTE_SLOT_MAPPING_KERNEL for the general (multi-request,
        # DCP/CP-aware) version this is a batch_size==1 simplification of.
        block_size = self.block_size
        block_idx = block_positions // block_size
        block_offset = block_positions % block_size
        physical_block_ids = cad.block_table_tensor[0, block_idx]
        slot_mapping = physical_block_ids * block_size + block_offset

        new_cad = CommonAttentionMetadata(
            query_start_loc=self.arange[:2] * num_tokens,
            query_start_loc_cpu=self.arange[:2].cpu() * num_tokens,
            seq_lens=torch.tensor(
                [seq_len + num_tokens], device=self.device, dtype=torch.int32
            ),
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=seq_len + num_tokens,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
        )

        # This proposer predicts one token per position from position 0 to
        # num_speculative_tokens - 1 (position i's logits predict the token
        # that would occupy position i + 1); the final block position's
        # logits are not used this round, matching the milestone-2 eager
        # decode loop's convention (see OrthrusForCausalLM.generate_with_
        # diffusion for the offline version of this same indexing).
        token_indices_to_sample = torch.arange(
            self.num_speculative_tokens, device=self.device
        )

        return num_tokens, token_indices_to_sample, new_cad
