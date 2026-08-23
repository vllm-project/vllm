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

Masking: within each request's proposed block, this now uses non-causal
(bidirectional) masking matching the reference implementation's
`generate_dual_pass_mask` in `modeling_orthrus.py`, reusing vLLM's
existing prefix-LM masking mechanism (FlexAttentionBackend's
`mm_req_doc_ranges` -> `get_prefix_lm_mask_mod`, normally used for
vision-language bidirectional token spans) rather than writing new
mask_mod code -- see `set_inputs_first_pass` and
`_create_draft_vllm_config`. Note that even the earlier plain-causal
version was already exactly lossless (vLLM's rejection sampling verifies
every proposed token against a real AR forward regardless of how the
draft produced it); this masking change is about matching the reference's
proposer *quality* (higher expected acceptance rate per round), not about
fixing a correctness issue. Reusing existing, presumably-tested masking
infrastructure lowers the risk here relative to writing a bespoke
mask_mod, but this specific combination (forcing FlexAttentionBackend
onto a draft model's KV-shared layers, with per-request doc ranges built
from a live block table) is still unvalidated against a running engine.
"""

import torch
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

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
        # replace() copies only init fields, and static_forward_context is
        # declared init=False -- so this preserves every compilation setting
        # (enforce_eager, cudagraph mode, ...) while giving the draft a
        # fresh, empty layer registry. Constructing a bare CompilationConfig()
        # instead would silently drop those settings.
        base = replace(base, compilation_config=replace(base.compilation_config))
        # Force FlexAttentionBackend for the draft's attn_diff layers so the
        # non-causal-within-block mask (see set_inputs_first_pass's
        # mm_req_doc_ranges below) is actually honored -- vLLM's other
        # paged backends (FlashAttention/Triton) only support causal or
        # sliding-window masks, not this hybrid causal-prefix +
        # bidirectional-region shape.
        return replace(
            base,
            attention_config=replace(
                base.attention_config, backend=AttentionBackendEnum.FLEX_ATTENTION
            ),
        )

    @override
    def _get_model(self) -> torch.nn.Module:
        # Mark this construction as the diffusion draft, so the model builds
        # its attn_diff layers and routes forward() through the diffusion
        # path. See building_diffusion_draft's docstring for why an explicit
        # context flag is needed rather than a VllmConfig identity check.
        from vllm.model_executor.models.orthrus import building_diffusion_draft

        with building_diffusion_draft():
            return super()._get_model()

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
            # copies frees most of its redundant weight memory; the AR
            # attention projections it also never uses are left in place
            # since they are comparatively small.
            if idx < len(target_layers):
                target_layer = target_layers[idx]
                layer.mlp = target_layer.mlp
                layer.input_layernorm = target_layer.input_layernorm
                layer.post_attention_layernorm = target_layer.post_attention_layernorm

        logger.info(
            "OrthrusProposer: wired %d draft layers to KV-share with the "
            "target and share its MLP/layernorm weights.",
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
            (batch_size, block_len), mask_token_id, dtype=torch.int32, device=self.device
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
        req_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(
            -1, block_len
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

        # Non-causal-within-block masking (matching the reference
        # implementation's diffusion pass, see generate_dual_pass_mask in
        # modeling_orthrus.py): each request's proposed block should attend
        # bidirectionally among its own new positions, on top of the usual
        # causal view of everything before it. Reusing vLLM's existing
        # prefix-LM masking mechanism (FlexAttentionBackend's mm_prefix_range
        # -> get_prefix_lm_mask_mod, normally used for vision-language
        # bidirectional token spans) does exactly this: it ORs a
        # bidirectional mask over each listed (start, end) logical position
        # range on top of the base causal mask. Requires attn_diff to be on
        # FlexAttentionBackend -- see _create_draft_vllm_config above.
        #
        # KNOWN LIMITATION: mm_req_doc_ranges is a plain Python dict of
        # per-request position ranges, so populating it forces one
        # device->host sync per step to read seq_lens. That is inherent to
        # this API (it is consumed on the host when building the FlexAttention
        # block mask), and it is the main reason this path is not yet
        # CUDA-graph-capturable. A first-class, tensor-native mask_mod for
        # this pattern would remove both limitations at once -- tracked on
        # the PR checklist alongside the design review of reusing this
        # mechanism at all.
        seq_lens_list = seq_lens.tolist()
        mm_req_doc_ranges = {
            req: [(seq_lens_list[req], seq_lens_list[req] + block_len - 1)]
            for req in range(batch_size)
        }

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
            causal=True,
            mm_req_doc_ranges=mm_req_doc_ranges,
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
