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
plain, vectorized PyTorch indexing supporting multi-request batches. This
piece is the highest-remaining-risk part of this whole effort: it computes
physical KV-cache slot indices from the block table using the standard
`physical_block_id * block_size + offset` formula (the same one vLLM's own
`_COMPUTE_SLOT_MAPPING_KERNEL` implements more generally, see
vllm/v1/worker/block_table.py, including DCP/CP-aware handling this
simplified version does not), but has NOT been validated against a running
engine due to persistent Modal-side infrastructure failures during testing
(see PR discussion) -- treat as an unverified, best-effort attempt for
review, not a confirmed-working implementation.

Masking: within each request's proposed block, this uses plain causal
masking (`causal=True`) rather than the reference implementation's
non-causal "attend everywhere in the block" scheme (see
`generate_dual_pass_mask` in the reference `modeling_orthrus.py`). This is
a deliberate simplification, not a shortcut on correctness: vLLM's
speculative-decode rejection sampling verifies every proposed token
against a real AR forward regardless of how the draft produced it, so a
weaker (causal) proposer is still exactly lossless -- it may just accept
fewer tokens per round than the paper's design. Matching the reference's
exact non-causal masking would need the target's own attention backend to
support a custom hybrid causal+bidirectional mask_mod (vLLM's
FlexAttentionBackend already supports the general mechanism -- see its
`get_prefix_lm_mask_mod` for a structurally similar causal-prefix +
bidirectional-region mask), which is a larger, separate change left for a
follow-up once this simpler version is confirmed working.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CompilationConfig
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
    def _create_draft_vllm_config(self) -> VllmConfig:
        """Give the draft build its own, isolated attention-layer registry.

        Both the target and this draft are the same OrthrusForCausalLM
        class with the same default layer-name prefixes (e.g.
        "model.layers.0.self_attn.attn"). The base class's
        _create_draft_vllm_config only replaces kernel_config, leaving
        compilation_config (and its static_forward_context registry, keyed
        purely by prefix string -- see Attention.__init__ in
        vllm/model_executor/layers/attention/attention.py) as the SAME
        shared object as the target's, causing a "Duplicate layer name"
        error when the draft's layers try to register under the same
        names. A fresh CompilationConfig gives the draft its own registry.
        Layer objects are still looked up via self.vllm_config (the
        target's, unaffected by this) wherever OrthrusProposer or vLLM's
        KV-cache-group setup needs to resolve kv_sharing_target_layer_name.
        """
        base = super()._create_draft_vllm_config()
        return replace(base, compilation_config=CompilationConfig())

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

        UNVALIDATED (see module docstring). Supports batch_size >= 1: every
        request in the batch proposes the same fixed-size block in this one
        forward, vectorized across requests (no per-request Python loop, no
        custom Triton kernel -- see the caveats below on what that trades
        away relative to vLLM's own general-purpose slot-mapping kernel).

        For each request, builds a block of ``num_speculative_tokens + 1``
        positions continuing directly after that request's current
        sequence: position 0 holds the last accepted/sampled token
        (``next_token_ids``), and the rest are filled with
        ``mask_token_id`` -- mirroring the reference implementation's
        diffusion-block construction (see ``modeling_orthrus.py``'s
        ``generate`` method), except here the block's tokens get real
        slot-mapped positions in the *target's* paged cache (via
        kv_sharing) rather than the reference's ephemeral, discarded KV.

        Known gaps versus a production implementation:
        - Every request is assumed to already be past prefill (pure
          decode-continuation positions); a request still in prefill this
          step is not specifically handled/excluded.
        - No padding/CUDA-graph-capture-size handling beyond what the base
          class's ``_determine_batch_execution_and_padding`` does generically
          for the returned ``num_tokens``.
        - Uses plain PyTorch indexing rather than vLLM's own
          ``_COMPUTE_SLOT_MAPPING_KERNEL`` (see
          vllm/v1/worker/block_table.py), which also handles decode context
          parallelism (DCP) and interleaved KV-cache layouts that this
          implementation does not.
        """
        batch_size = cad.batch_size()
        mask_token_id = self.vllm_config.model_config.hf_config.mask_token_id
        block_len = self.num_speculative_tokens + 1
        num_tokens = batch_size * block_len

        # --- input_ids: [batch_size, block_len] -> flattened, row-major
        # (one request's full block before the next request's). ---
        block_ids = torch.full(
            (batch_size, block_len), mask_token_id, dtype=torch.int32, device=self.device
        )
        block_ids[:, 0] = next_token_ids[:batch_size]
        self.input_ids[:num_tokens] = block_ids.reshape(-1)

        # --- positions: request i's block covers
        # [seq_len_i, seq_len_i + block_len). ---
        seq_lens = cad.seq_lens[:batch_size].to(torch.int64)
        block_positions = seq_lens.unsqueeze(1) + torch.arange(
            block_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)  # [batch_size, block_len]
        self._set_positions(num_tokens, block_positions.reshape(-1))

        # --- slot_mapping: physical slot for each new position, per its
        # own request's block table. Standard paged-attention convention
        # (block_table[req, pos // block_size] * block_size + pos %
        # block_size) -- see vllm/v1/worker/block_table.py's
        # _COMPUTE_SLOT_MAPPING_KERNEL for the general (DCP/CP-aware)
        # version this is a simplification of. ---
        block_size = self.block_size
        block_idx = block_positions // block_size  # [batch_size, block_len]
        block_offset = block_positions % block_size
        req_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(
            -1, block_len
        )
        physical_block_ids = cad.block_table_tensor[req_idx, block_idx]
        slot_mapping = (physical_block_ids * block_size + block_offset).reshape(-1)

        new_seq_lens = (seq_lens + block_len).to(torch.int32)
        new_query_start_loc = self.arange[: batch_size + 1] * block_len

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc.cpu(),
            seq_lens=new_seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=block_len,
            max_seq_len=int(new_seq_lens.max().item()),
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
        )

        # This proposer predicts one token per position from block-local
        # position 0 to num_speculative_tokens - 1 (position i's logits
        # predict the token that would occupy position i + 1); each
        # request's final block position's logits are not used this round,
        # matching the milestone-2 eager decode loop's convention (see
        # OrthrusForCausalLM.generate_with_diffusion for the offline
        # version of this same indexing). Indices are offset per request
        # since the logits tensor is flattened in the same [batch_size,
        # block_len] row-major order as input_ids above.
        token_indices_to_sample = (
            torch.arange(batch_size, device=self.device).unsqueeze(1) * block_len
            + torch.arange(self.num_speculative_tokens, device=self.device).unsqueeze(0)
        ).reshape(-1)

        return num_tokens, token_indices_to_sample, new_cad
