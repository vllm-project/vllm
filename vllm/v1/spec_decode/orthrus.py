# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXPERIMENTAL Orthrus diffusion-mode proposer for speculative decoding.

Status: EXPERIMENTAL and not yet validated end-to-end (see the discussion
on https://github.com/vllm-project/vllm/pull/44792). It IS wired into the
engine -- ``speculative_config={"method": "orthrus", ...}`` constructs
this proposer and reaches its ``propose()`` path -- but has not completed
a successful generate() run, and has not been run against vLLM's CI,
multi-GPU, or CUDA graph capture.

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
simplified version does not). Treat as unverified until a successful
generate() run is posted on the PR.

Masking: plain unmasked (fully bidirectional) attention across the
target's entire cached AR history plus the whole new block -- no causal
restriction, no per-position mask at all. This matches what the
reference implementation (`modeling_orthrus.py`) actually runs at
*inference*: `generate_dual_pass_mask`'s FlexAttention block mask is
built and used only under `if self.training`, and the reference's own
`generate()` calls the diffusion forward with no `attention_mask` and
`is_causal=False` -- i.e. the full bidirectional-everything path, not
the more restrictive training-time mask.

Two earlier attempts at this were both *more* restrictive than that and
neither fixed acceptance rate: plain causal-only, then a causal-history +
bidirectional-within-block prefix-LM mask built by forcing
FlexAttentionBackend onto the draft and reusing `mm_req_doc_ranges` ->
`get_prefix_lm_mask_mod` (this needed a KV-cache-layout compatibility fix
to even run -- see git history for that). Running the reference
implementation directly as a baseline (same checkpoint, same block
length) got 39-82% acceptance against ~2-4% for both of those schemes,
which is what motivated re-reading the reference's actual inference path
instead of just its training-time mask helper. Plain `causal=False` on
the draft's `CommonAttentionMetadata` reproduces that unmasked behavior
without needing FlexAttention at all, so the backend-forcing and
`mm_req_doc_ranges` plumbing are gone too.

Even a fully wrong mask would still be exactly lossless (vLLM's
rejection sampling verifies every proposed token against a real AR
forward regardless of how the draft produced it); masking only affects
proposer *quality* (acceptance rate per round), not correctness.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

logger = init_logger(__name__)

# Namespaces the draft's layer names apart from the target's in the shared
# compilation_config.static_forward_context registry (both are the same
# OrthrusForCausalLM class, so they would otherwise collide).
DRAFT_PREFIX = "orthrus_diffusion_draft"


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
    def model_returns_tuple(self) -> bool:
        """Orthrus' diffusion forward returns a single hidden-states tensor.

        The base class assumes a ``(last_hidden_states, hidden_states)`` pair
        for every method outside a small allowlist, and unpacking a plain
        ``[num_tokens, hidden_size]`` tensor into two names raises
        "too many values to unpack" on the first propose() call.
        """
        return False

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        """Keep the draft on the target's attention backend.

        The draft's attention layers KV-share the *target's* cache tensors,
        so both must agree on the KV cache layout. The base class resets
        attention_config.backend to None for draft models, which can let
        them pick a different backend than the target -- same hazard
        Gemma4Proposer documents ("FLASH_ATTN ... cannot handle KV-shared
        cache"). Carry the target's backend through explicitly.

        No FlexAttention needed here: the masking this proposer wants
        (plain unmasked/bidirectional, see the module docstring) is just
        `causal=False` on the block's `CommonAttentionMetadata`, which
        `FlashAttentionBackend` already supports natively
        (`supports_non_causal()`).
        """
        base = super()._create_draft_vllm_config()
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is None:
            return base
        return replace(
            base,
            attention_config=replace(base.attention_config, backend=target_backend),
        )

    # initialize_cudagraph_keys: no longer overridden. It used to force
    # CUDAGraphMode.NONE unconditionally, because forward_diffusion_paged
    # was called directly (bypassing the compiled __call__ entry point
    # @support_torch_compile instruments on OrthrusModel) -- PIECEWISE
    # capture would have set up a dispatcher for graphs that forward never
    # actually populated. OrthrusModel.forward now dispatches to
    # forward_diffusion_paged internally instead of being bypassed from
    # the outer LM's forward, so both paths go through the same compiled
    # entry point and the base class's real PIECEWISE-when-eligible logic
    # (SpecDecodeBaseProposer.initialize_cudagraph_keys) applies. UNTESTED
    # under actual capture as of this change -- forward_diffusion_paged's
    # data-dependent slot_mapping lookups and the manual KV-cache indexing
    # in _force_write_diffusion_kv have never run under CUDA graph replay.

    @override
    def _get_model(self) -> torch.nn.Module:
        """Build the draft under a distinct prefix and the draft-mode flag.

        The draft is the same OrthrusForCausalLM class as the target, so
        with the default empty prefix its attention layers would register
        under names the target already claimed (e.g.
        "model.layers.0.self_attn.attn") in the shared
        compilation_config.static_forward_context, raising "Duplicate layer
        name". Building under DRAFT_PREFIX namespaces them instead.

        Note this registry must stay *shared* with the target's: the base
        class discovers draft layers by diffing that same registry (see
        load_model's _draft_attn_layer_names), and per-layer attention
        metadata is keyed by these names at runtime. Giving the draft its
        own CompilationConfig would hide its layers from that discovery and
        surface later as a KeyError on the first diffusion forward.

        building_diffusion_draft() additionally makes the model build its
        attn_diff layers and route forward() through the diffusion path --
        see its docstring for why a context flag is needed rather than a
        VllmConfig identity check.
        """
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.models.orthrus import building_diffusion_draft

        draft_vllm_config = self._create_draft_vllm_config()
        with building_diffusion_draft(), set_model_tag("orthrus_diffusion_draft"):
            return get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
                prefix=DRAFT_PREFIX,
            )

    @override
    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        self._setup_orthrus_kv_sharing(target_model)
        logger.info(
            "OrthrusProposer: discovered %d draft attention layers: %s",
            len(self._draft_attn_layer_names),
            sorted(self._draft_attn_layer_names),
        )

    @override
    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        logger.info(
            "OrthrusProposer: built %d draft attention group(s); "
            "layers with metadata: %s",
            len(self.draft_attn_groups),
            sorted(
                name for group in self.draft_attn_groups for name in group.layer_names
            ),
        )

    def _setup_orthrus_kv_sharing(self, target_model: torch.nn.Module) -> None:
        """Map each draft layer's diffusion attention to its same-index
        target layer's AR attention, so a diffusion forward reads the
        target's real, already-populated paged KV cache."""
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return
        if not (
            hasattr(target_model, "model") and hasattr(target_model.model, "layers")
        ):
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

        target_layers = target_model.model.layers
        for idx, layer in enumerate(self.model.model.layers):
            target_layer_name = f"{target_prefix}.{idx}.self_attn.attn"

            attn_diff = getattr(layer.self_attn, "attn_diff", None)
            if attn_diff is not None:
                attn_diff.kv_sharing_target_layer_name = target_layer_name

            # The draft's *AR* attention layer is never called (its
            # forward() routes through the diffusion path), but it is still
            # a registered Attention module -- so without a sharing target
            # it would be handed a full KVCacheSpec of its own and allocate
            # an entire unused second KV cache. Point it at the same target
            # layer so it allocates nothing.
            attn_ar = getattr(layer.self_attn, "attn", None)
            if attn_ar is not None:
                attn_ar.kv_sharing_target_layer_name = target_layer_name

            # Share the modules the diffusion forward uses but does not
            # specialize (MLP and both layernorms are identical to the
            # target's -- Orthrus only adds *_diff attention parameters on
            # top of a frozen backbone). Dropping the draft's duplicate
            # copies frees most of its redundant weight memory. The AR
            # attention projections (qkv_proj/o_proj) are never called by
            # the diffusion forward either -- forward_diffusion_paged only
            # reaches qkv_proj_diff/o_proj_diff/attn_diff -- so they are
            # re-pointed at the target's the same way.
            if idx < len(target_layers):
                target_layer = target_layers[idx]
                layer.mlp = target_layer.mlp
                layer.input_layernorm = target_layer.input_layernorm
                layer.post_attention_layernorm = target_layer.post_attention_layernorm
                target_self_attn = target_layer.self_attn
                if hasattr(layer.self_attn, "qkv_proj") and hasattr(
                    target_self_attn, "qkv_proj"
                ):
                    layer.self_attn.qkv_proj = target_self_attn.qkv_proj
                if hasattr(layer.self_attn, "o_proj") and hasattr(
                    target_self_attn, "o_proj"
                ):
                    layer.self_attn.o_proj = target_self_attn.o_proj

        # embed_tokens/lm_head are also never specialized by the diffusion
        # path (embed_input_ids and compute_logits both go through the same
        # weights as the target) -- share them too instead of keeping the
        # draft's own duplicate copies from checkpoint load.
        if hasattr(self.model.model, "embed_tokens") and hasattr(
            target_model.model, "embed_tokens"
        ):
            self.model.model.embed_tokens = target_model.model.embed_tokens
        if hasattr(self.model, "lm_head") and hasattr(target_model, "lm_head"):
            self.model.lm_head = target_model.lm_head

        logger.info(
            "OrthrusProposer: wired %d draft layers to KV-share with the "
            "target and share its MLP/layernorm/AR-projection/embedding "
            "weights.",
            len(self.model.model.layers),
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
        # Resolved by the base class from the *draft* config at init time
        # (see _init_parallel_drafting_params, enabled for "orthrus" via
        # SpeculativeConfig.parallel_drafting), which also raises a clear
        # error up front if the checkpoint has no mask_token_id -- rather
        # than failing here on the first decode step.
        mask_token_id = self.parallel_drafting_token_id
        block_len = self.num_speculative_tokens + 1
        num_tokens = batch_size * block_len

        # cad.seq_lens can be an optimistic upper bound that assumes every
        # draft token from the *previous* round was accepted (see
        # CommonAttentionMetadata.seq_lens_cpu_upper_bound's docstring).
        # Subtract num_rejected_tokens_gpu to get the real continuation
        # point -- otherwise, starting from round 2 of speculative decoding
        # (after any rejection at all), every position/slot computed below
        # would be wrong, silently drifting away from the request's actual
        # sequence length and writing into the wrong KV-cache slots.
        # Mirrors DFlashProposer.set_inputs_first_pass's effective_seq_lens.
        effective_seq_lens = cad.seq_lens[:batch_size]
        if num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # --- input_ids: [batch_size, block_len] -> flattened, row-major
        # (one request's full block before the next request's). ---
        block_ids = torch.full(
            (batch_size, block_len),
            mask_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        block_ids[:, 0] = next_token_ids[:batch_size]
        self.input_ids[:num_tokens] = block_ids.reshape(-1)

        # --- positions: request i's block covers
        # [seq_len_i, seq_len_i + block_len). ---
        seq_lens = effective_seq_lens.to(torch.int64)
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
        # Clamp before block-table lookup and mask out slots for any
        # position that would exceed max_model_len: an un-clamped
        # block_idx could read arbitrary GPU memory as a physical block id
        # once a proposed block runs past the end of a request's allocated
        # blocks, which would then get used for KV-cache reads/writes --
        # potentially touching another request's data. Same pattern as
        # compute_new_slot_mapping / the Eagle proposer's clamping.
        block_size = self.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        clamped_positions = torch.clamp(block_positions, max=max_model_len - 1)
        block_idx = clamped_positions // block_size  # [batch_size, block_len]
        block_offset = clamped_positions % block_size
        req_idx = (
            torch.arange(batch_size, device=self.device)
            .unsqueeze(1)
            .expand(-1, block_len)
        )
        physical_block_ids = cad.block_table_tensor[req_idx, block_idx]
        slot_mapping = (physical_block_ids * block_size + block_offset).reshape(-1)
        exceeds_max_len = (block_positions >= max_model_len).reshape(-1)
        slot_mapping = slot_mapping.masked_fill(exceeds_max_len, PADDING_SLOT_ID)

        new_seq_lens = (seq_lens + block_len).to(torch.int32)
        new_query_start_loc = self.arange[: batch_size + 1] * block_len
        # Build the CPU copy from the host-side arange rather than a .cpu()
        # of the device tensor, to avoid a device->host sync per step
        # (same approach as DFlashProposer.set_inputs_first_pass).
        new_query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * block_len
        )

        # Plain unmasked (fully bidirectional) attention, matching what the
        # reference implementation actually runs at inference (see module
        # docstring): every new block position attends to everything --
        # the target's whole cached AR history plus every other position in
        # this same new block, including "later" ones. No per-request doc
        # ranges or custom mask_mod needed, unlike two earlier attempts at
        # this (plain causal, then a causal-history + bidirectional-block
        # prefix-LM mask) -- both were more restrictive than the reference
        # and neither fixed acceptance rate.
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=block_len,
            # Upper bound is sufficient here and avoids another sync.
            max_seq_len=cad.max_seq_len + block_len,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=slot_mapping,
            causal=False,
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
