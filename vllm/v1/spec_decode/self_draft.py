# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-Draft Parallel Decoding (PaRD-style) proposer.

The target model itself acts as the drafter: ``num_speculative_tokens``
mask tokens are appended to each request and the model fills them in
with a single forward pass; the standard rejection sampler then verifies
a contiguous prefix of matches.

Reference: "PaRD: A Parallel Decoding Framework..." (chain layout, K
mask tokens per request, drafter shares weights with the target).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.utils import compute_new_slot_mapping

logger = init_logger(__name__)


class SelfDraftProposer(DraftModelProposer):
    """Parallel-decoding proposer where the drafter is the target model.

    This subclass exists to (a) bypass the multimodal / M-RoPE /
    vocab-size / TP guards in the base proposer chain (they assume a
    separate drafter checkpoint) and (b) build drafter inputs that
    reuse the target's KV cache instead of re-running the prompt.
    """

    # ------------------------------------------------------------------ #
    # Guards bypassed by self_draft. ``_raise_if_*`` methods are invoked
    # by ``SpecDecodeBaseProposer.__init__`` via MRO, so these overrides
    # must exist before ``super().__init__`` runs.
    # ------------------------------------------------------------------ #

    @override
    def _raise_if_multimodal(self) -> None:
        # self_draft natively supports multimodal targets.
        return

    @override
    def _raise_if_mrope(self) -> None:
        # 3-D positions are handled by ``set_inputs_first_pass`` below.
        return

    @override
    def _init_parallel_drafting_params(self) -> None:
        """Resolve the mask token id for parallel drafting.

        The mask token id must be supplied via
        ``SpeculativeConfig.parallel_drafting_mask_token_id`` (typically
        forwarded by the engine from a user-facing config or CLI flag).
        We intentionally do not sniff arbitrary ``hf_config`` attributes
        here so the contract is explicit on the public API surface.
        """
        override_id = getattr(
            self.speculative_config, "parallel_drafting_mask_token_id", None
        )
        if override_id is None:
            raise ValueError(
                "SelfDraftProposer requires an explicit mask-token id. "
                "Set `SpeculativeConfig.parallel_drafting_mask_token_id` "
                "(e.g. via `speculative_config={'method': 'self_draft', "
                "'parallel_drafting_mask_token_id': <id>, ...}`)."
            )
        self.parallel_drafting_token_id = int(override_id)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        spec = vllm_config.speculative_config
        assert spec is not None, "Speculative config must be set"
        if not spec.parallel_drafting:
            raise ValueError(
                "SelfDraftProposer requires `SpeculativeConfig.parallel_drafting=True`."
            )
        if runner is None:
            raise ValueError(
                "SelfDraftProposer needs a reference to the GPU model "
                "runner to reuse the target model module."
            )

        # ``SpecDecodeBaseProposer.__init__`` invokes the overridden
        # ``_raise_if_*`` / ``_init_parallel_drafting_params`` above.
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

        self._runner = runner
        logger.debug(
            "SelfDraftProposer initialized: mask_token_id=%d, "
            "supports_mm_inputs=%s, uses_mrope=%s",
            self.parallel_drafting_token_id,
            self.supports_mm_inputs,
            self.uses_mrope,
        )

    # ``DraftModelProposer.__init__`` runs two extra checks
    # (``_raise_if_vocab_size_mismatch`` / ``_raise_if_draft_tp_mismatch``)
    # that only apply to a separate drafter checkpoint. For self_draft
    # the drafter IS the target, so both are trivially satisfied.
    @override
    def _raise_if_vocab_size_mismatch(self) -> None:
        return

    @override
    def _raise_if_draft_tp_mismatch(self) -> None:
        return

    def _resolve_target_model(self) -> nn.Module:
        model = getattr(self._runner, "model", None)
        if not isinstance(model, nn.Module):
            raise AttributeError(
                "SelfDraftProposer could not locate the target model on "
                "the model runner (`runner.model`)."
            )
        return model

    @override
    def load_model(self, target_model: nn.Module) -> None:
        """Reuse the target model as the drafter.

        ``self._draft_attn_layer_names`` is normally derived as
        ``all_attn_layers - target_attn_layer_names``; for self_draft
        that difference is empty. We instead use the full attention
        layer set; ``validate_same_kv_cache_group`` then filters out
        layers without a kv-cache group (e.g. vision tower layers).

        Note: we intentionally do not cache ``self.model`` here. The
        GPU model runner wraps its model in ``CUDAGraphWrapper`` *after*
        ``drafter.load_model`` runs, so caching would bypass the wrapper
        on every drafter forward and break FULL cudagraph replay. The
        ``self.model`` property below always reads ``runner.model``.
        """
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Triggers subclass-specific side effects (logging, etc.).
        _ = self._get_model()
        self._draft_attn_layer_names = set(all_attn_layers.keys())

        # Multimodal targets need ``embed_input_ids`` available and an
        # image_token_index on the drafter's config for downstream
        # multimodal-aware code paths.
        if self.supports_mm_inputs:
            if not hasattr(self.model, "embed_input_ids"):
                raise AttributeError(
                    "SelfDraftProposer requires the target model to "
                    "expose `embed_input_ids` for multimodal inputs."
                )
            image_token_idx = getattr(
                target_model.config,
                "image_token_index",
                getattr(target_model.config, "image_token_id", None),
            )
            if image_token_idx is not None:
                self.model.config.image_token_index = image_token_idx

    @override
    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        """Restrict the draft layer set to layers that belong to a
        kv-cache group (filters out non-KV-cached layers such as
        vision tower blocks) and assert they share a single group.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id

        self._draft_attn_layer_names = {
            n for n in self._draft_attn_layer_names if n in kv_cache_groups
        }
        group_ids = {kv_cache_groups[n] for n in self._draft_attn_layer_names}
        assert len(group_ids) <= 1, (
            f"SelfDraft: all language-side attention layers must belong "
            f"to the same kv cache group, got {group_ids}"
        )

    # ``self.model`` reads through ``runner.model`` so the drafter
    # always sees the runner's (possibly CUDAGraphWrapper-wrapped)
    # current model reference. See ``load_model`` for rationale.
    @property
    def model(self) -> nn.Module:
        return getattr(self._runner, "model", None)  # type: ignore[return-value]

    def _get_persist_block_table(self, num_reqs: int) -> torch.Tensor | None:
        """Return a stable view into the runner's persistent block table.

        Caches the full-size device tensor on first call so subsequent
        slices ``[:num_reqs]`` share the same ``data_ptr`` — required
        for FULL cudagraph replay.
        """
        full = getattr(self, "_persist_block_table_full", None)
        if full is None:
            input_batch = getattr(self._runner, "input_batch", None)
            if input_batch is None:
                return None
            bt = input_batch.block_table[0]
            full = bt.get_device_tensor(self.max_batch_size)
            self._persist_block_table_full = full
        return full[:num_reqs]

    @override
    def _get_model(self) -> nn.Module:
        target_model = self._resolve_target_model()
        logger.debug(
            "SelfDraftProposer reusing target model module (%s) as the "
            "drafter; no extra weights loaded.",
            target_model.__class__.__name__,
        )
        return target_model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Drafter IS the target — embeddings are already shared.
        return

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        return

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        # Drafter shares the target's VllmConfig.
        return self.vllm_config

    @override
    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ):
        """Build attention metadata for the drafter forward.

        FA3 (``FlashAttentionMetadataBuilder``) owns a long-lived
        ``scheduler_metadata`` buffer that is written in place by
        ``build_for_cudagraph_capture`` (full build). ``fast_build=True``
        — used by the default ``build_for_drafting`` — does not refresh
        this buffer, so a FULL-captured graph would replay a stale
        schedule (computed from capture-time placeholder ``seq_lens``).
        For builders that participate in FULL cudagraph AOT scheduling,
        we route through ``build_for_cudagraph_capture`` to refresh the
        in-place schedule with the actual ``seq_lens`` of this step.

        TODO: replace this override once FA3 exposes a
        ``force_schedule_refresh`` flag on ``build_for_drafting``.
        """
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            needs_schedule_refresh = (
                getattr(builder, "use_full_cuda_graph", False)
                and getattr(builder, "scheduler_metadata", None) is not None
                and draft_index == 0
            )
            if needs_schedule_refresh:
                attn_metadata = builder.build_for_cudagraph_capture(
                    common_attn_metadata=common_attn_metadata
                )
            else:
                attn_metadata = builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=draft_index,
                )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

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
        """Build drafter inputs that reuse the target's KV cache.

        Per-request layout in the expanded batch:
            ``[bonus | mask_token × (K - 1)]``  (K tokens)
        with positions ``[S, S+1, ..., S+K-1]`` where ``S`` is the
        post-acceptance physical sequence length.

        The prompt is not re-run on the drafter — its KV is already
        cached by the target. For multi-step spec decoding,
        ``num_rejected_tokens_gpu`` accounts for tokens whose KV was
        written but rejected.

        For M-RoPE VLMs the drafter extends all three RoPE dims by +1
        per step (text-continuation semantics), based on the M-RoPE
        position of each request's last accepted token in
        ``target_positions``.

        Outputs (capture-time alignment): all attention-metadata fields
        the per-layer metadata references must reside in persistent
        buffers (``self._persist_*``) so that propose-time addresses
        match the addresses baked into the captured cudagraphs.
        """
        assert self.is_rejected_token_mask is not None
        assert self.is_masked_token_mask is not None
        assert self.parallel_drafting, (
            "SelfDraftProposer requires parallel_drafting=True."
        )
        assert not self.pass_hidden_states_to_model, (
            "SelfDraftProposer derives from DraftModelProposer which "
            "sets pass_hidden_states_to_model=False; hidden-state "
            "plumbing is intentionally unsupported."
        )

        batch_size = cad.batch_size()
        K = self.extra_slots_per_request  # == num_speculative_tokens
        device = self.input_ids.device

        # Physical sequence length after acceptance, used as the start
        # position of the K new drafter tokens in each request's KV.
        seq_lens_gpu = cad.seq_lens  # [batch], int32
        if num_rejected_tokens_gpu is not None:
            seq_lens_after_accept = seq_lens_gpu - num_rejected_tokens_gpu
        else:
            seq_lens_after_accept = seq_lens_gpu
        seq_lens_after_accept = seq_lens_after_accept.to(torch.int64)

        total_num_output_tokens = batch_size * K

        self.is_rejected_token_mask[:total_num_output_tokens].zero_()
        self.is_masked_token_mask[:total_num_output_tokens].zero_()

        # Input ids: per-request ``[bonus, mask, mask, ...]``.
        input_ids_slice = self.input_ids[:total_num_output_tokens]
        input_ids_slice.fill_(self.parallel_drafting_token_id)
        bonus_positions_in_flat = torch.arange(
            0, total_num_output_tokens, K, device=device, dtype=torch.long
        )
        input_ids_slice[bonus_positions_in_flat] = next_token_ids.to(
            input_ids_slice.dtype
        )
        # Mark mask slots (every slot except the per-request bonus).
        mask_flag = torch.ones(total_num_output_tokens, dtype=torch.bool, device=device)
        mask_flag[bonus_positions_in_flat] = False
        self.is_masked_token_mask[:total_num_output_tokens] = mask_flag

        # Physical KV positions: ``[S, S+1, ..., S+K-1]`` per request.
        k_offsets = torch.arange(K, device=device, dtype=torch.int64)
        pos_per_slot = (seq_lens_after_accept[:, None] + k_offsets[None, :]).reshape(
            -1
        )  # [batch*K], int64

        if self.uses_mrope:
            # M-RoPE VLMs: physical KV positions are NOT valid for
            # text-side rotary positions because vision tokens expand
            # to many KV slots with a compressed text-dim position. We
            # extend each request's last-accepted M-RoPE position by
            # +1..+K, mirroring the target's text-continuation update.
            assert target_positions.dim() == 2 and target_positions.shape[0] == 3, (
                "M-RoPE self-draft expects target_positions with shape "
                f"[3, num_tokens], got {tuple(target_positions.shape)}"
            )

            query_end_idx = cad.query_start_loc[1:].to(torch.int64) - 1
            if num_rejected_tokens_gpu is not None:
                last_accepted_idx = query_end_idx - num_rejected_tokens_gpu.to(
                    torch.int64
                )
            else:
                last_accepted_idx = query_end_idx

            base_mrope = target_positions.index_select(
                dim=1, index=last_accepted_idx
            )  # [3, batch]
            k_offsets_for_mrope = (k_offsets + 1).to(torch.int64)
            drafter_mrope = (
                base_mrope[:, :, None] + k_offsets_for_mrope[None, None, :]
            ).reshape(3, -1)  # [3, batch*K]
            self.mrope_positions[:, :total_num_output_tokens] = drafter_mrope

            # 1-D physical positions are still needed for slot_mapping.
            if not hasattr(self, "positions") or self.positions is None:
                self.positions = torch.zeros(
                    self.max_num_tokens + 1,
                    dtype=torch.int64,
                    device=device,
                )
            self.positions[:total_num_output_tokens] = pos_per_slot
        else:
            self.positions[:total_num_output_tokens] = pos_per_slot.to(
                self.positions.dtype
            )

        token_indices_to_sample = torch.arange(
            total_num_output_tokens, dtype=torch.int32, device=device
        )

        # Route attention-metadata fields through persistent buffers so
        # propose- and capture-time addresses match (see
        # ``initialize_cudagraph_keys`` for buffer allocation).
        if not hasattr(self, "_persist_query_start_loc"):
            # No persistent buffers (e.g. enforce_eager). Allocate fresh.
            new_query_start_loc_cpu = torch.arange(
                0,
                (batch_size + 1) * K,
                K,
                dtype=cad.query_start_loc_cpu.dtype,
            )
            new_query_start_loc = new_query_start_loc_cpu.to(
                device=cad.query_start_loc.device
            )
            new_seq_lens = (seq_lens_after_accept + K).to(cad.seq_lens.dtype)
        else:
            qsl_buf = self._persist_query_start_loc
            qsl_cpu_buf = self._persist_query_start_loc_cpu
            sl_buf = self._persist_seq_lens
            assert qsl_buf.dtype == torch.int32
            torch.arange(
                0,
                (batch_size + 1) * K,
                K,
                out=qsl_buf[: batch_size + 1],
            )
            qsl_cpu_buf[: batch_size + 1] = torch.arange(
                0, (batch_size + 1) * K, K, dtype=qsl_cpu_buf.dtype
            )
            sl_buf[:batch_size] = (seq_lens_after_accept + K).to(sl_buf.dtype)
            new_query_start_loc = qsl_buf[: batch_size + 1]
            new_query_start_loc_cpu = qsl_cpu_buf[: batch_size + 1]
            new_seq_lens = sl_buf[:batch_size]

        # ``compute_new_slot_mapping`` consumes the drafter query layout.
        drafter_cad_for_slot = cad.replace(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens,
            num_actual_tokens=total_num_output_tokens,
            max_query_len=K,
            max_seq_len=int(cad.max_seq_len) + K,
        )

        new_slot_mapping = compute_new_slot_mapping(
            cad=drafter_cad_for_slot,
            new_positions=pos_per_slot,
            is_rejected_token_mask=self.is_rejected_token_mask[
                :total_num_output_tokens
            ],
            block_size=self.block_size,
            # ``naive_query_lens`` on drafter_cad_for_slot is already K.
            num_new_tokens=0,
            max_model_len=self.max_model_len,
        )

        # Copy slot_mapping into the persistent buffer to keep the
        # captured graph's address stable.
        if hasattr(self, "_persist_slot_mapping"):
            sm_buf = self._persist_slot_mapping
            assert new_slot_mapping.dtype == sm_buf.dtype, (
                f"slot_mapping dtype mismatch: persistent buffer is "
                f"{sm_buf.dtype} but compute_new_slot_mapping produced "
                f"{new_slot_mapping.dtype}"
            )
            sm_buf[:total_num_output_tokens].copy_(new_slot_mapping)
            new_slot_mapping = sm_buf[:total_num_output_tokens]

        block_table_tensor_persist = self._get_persist_block_table(batch_size)
        if block_table_tensor_persist is not None:
            new_cad_block_table = block_table_tensor_persist
        else:
            new_cad_block_table = drafter_cad_for_slot.block_table_tensor

        new_cad = drafter_cad_for_slot.replace(
            slot_mapping=new_slot_mapping,
            block_table_tensor=new_cad_block_table,
        )

        return total_num_output_tokens, token_indices_to_sample, new_cad

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Register FULL cudagraph keys for the self-draft drafter.

        Every drafter forward has ``num_tokens = batch_size * K`` and
        is uniform-decode (each request contributes exactly K tokens).
        That makes the drafter shape space FULL-graph-capturable; the
        base class otherwise restricts spec proposers to PIECEWISE
        because EAGLE-style drafters have non-uniform query_len.

        Capture keys are registered at exact ``num_tokens = bs * K``
        for every ``bs ∈ [1, max_num_seqs]``; PIECEWISE keys (seeded by
        the base class) act as a fallback for shapes outside that set.

        TODO: align with ``adjust_cudagraph_sizes_for_spec_decode``
        once it supports drafter shapes that are not multiples of
        ``1 + K``.
        """
        from vllm.forward_context import BatchDescriptor

        dispatcher = self.cudagraph_dispatcher

        if self.speculative_config.enforce_eager:
            dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
            return
        if cudagraph_mode == CUDAGraphMode.NONE:
            dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
            return

        K = self.num_speculative_tokens
        # Pin the descriptor padding math to K before seeding keys.
        dispatcher.uniform_decode_query_len = K

        # Seed PIECEWISE keys / bookkeeping via the base behavior.
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE)

        if cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL and (
            cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
        ):
            return

        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        batch_sizes = list(range(1, max_num_seqs + 1))
        self._self_draft_full_batch_sizes: list[int] = batch_sizes

        # TODO: use a public ``dispatcher.get_lora_cases()`` once
        # exposed; the private accessor is used to stay aligned with
        # the dispatcher's current lora-aware key registration.
        lora_cases = dispatcher._get_lora_cases()
        for bs in batch_sizes:
            for num_active_loras in lora_cases:
                batch_desc = BatchDescriptor(
                    num_tokens=bs * K,
                    num_reqs=bs,
                    uniform=True,
                    has_lora=num_active_loras > 0,
                    num_active_loras=num_active_loras,
                )
                dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, batch_desc)

        # Allow FULL through dispatch().
        dispatcher.cudagraph_mode = cudagraph_mode

        logger.info(
            "SelfDraftProposer registered FULL cudagraph keys at "
            "num_reqs=[1..%d] (num_tokens = num_reqs * K=%d); lora_cases=%s",
            max_num_seqs,
            K,
            lora_cases,
        )

        # Allocate persistent buffers backing attention metadata so the
        # device addresses captured during graph capture remain valid
        # at propose time. See ``set_inputs_first_pass`` for the
        # consumers; see ``capture_drafter_cudagraph`` for the writer.
        max_bs_seq = self.max_batch_size
        max_drafter_tokens = max_bs_seq * K
        device = self.input_ids.device
        if not hasattr(self, "_persist_query_start_loc"):
            self._persist_query_start_loc = torch.zeros(
                max_bs_seq + 1, dtype=torch.int32, device=device
            )
            self._persist_query_start_loc_cpu = torch.zeros(
                max_bs_seq + 1, dtype=torch.int32
            )
            self._persist_seq_lens = torch.zeros(
                max_bs_seq, dtype=torch.int32, device=device
            )
            self._persist_seq_lens_cpu = torch.zeros(max_bs_seq, dtype=torch.int32)
            self._persist_slot_mapping = torch.zeros(
                max_drafter_tokens, dtype=torch.int64, device=device
            )
            self._persist_block_table_full: torch.Tensor | None = None

    @override
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ):
        """Drafter dispatch tailored to ``num_tokens = bs * K``.

        The generic dispatcher pads via ``_bs_to_padded_graph_size``,
        which is derived from the target's spec-decode-adjusted capture
        sizes (multiples of ``1 + K``). Those sizes never align with
        the drafter's exact ``bs * K`` shape, so we hit a registered
        FULL key directly when possible and otherwise fall back to a
        non-uniform dispatch (PIECEWISE / NONE) — staying clear of the
        uniform-decode padding assert.
        """
        from vllm.config import CUDAGraphMode as _CGM
        from vllm.forward_context import BatchDescriptor
        from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

        K = self.num_speculative_tokens
        dispatcher = self.cudagraph_dispatcher

        cudagraph_mode: _CGM | None = None
        batch_desc: BatchDescriptor | None = None

        registered_bs = getattr(self, "_self_draft_full_batch_sizes", None)
        if (
            use_cudagraphs
            and dispatcher.keys_initialized
            and dispatcher.cudagraph_mode != _CGM.NONE
            and dispatcher.cudagraph_mode.has_mode(_CGM.FULL)
            and registered_bs
            and num_tokens > 0
            and num_tokens % K == 0
            and (num_tokens // K) in registered_bs
        ):
            num_reqs = num_tokens // K
            candidate = BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform=True,
                has_lora=False,
                num_active_loras=0,
            )
            if candidate in dispatcher.cudagraph_keys[_CGM.FULL]:
                cudagraph_mode = _CGM.FULL
                batch_desc = candidate

        if cudagraph_mode is None:
            # PIECEWISE / NONE fallback. uniform_decode=False avoids the
            # padding assert on shapes that are not multiples of K.
            cudagraph_mode, batch_desc = dispatcher.dispatch(
                num_tokens,
                uniform_decode=False,
                valid_modes=({_CGM.NONE} if not use_cudagraphs else None),
            )

        num_tokens_padded = batch_desc.num_tokens

        # DP coordination, mirrored from the base class.
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for self_draft"

            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                cudagraph_mode, batch_desc = dispatcher.dispatch(
                    num_tokens_padded,
                    uniform_decode=False,
                    valid_modes={_CGM(synced_cudagraph_mode)},
                )
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp

    @torch.inference_mode()
    def capture_drafter_cudagraph(self) -> None:
        """Capture one FULL cudagraph per registered drafter shape.

        ``GPUModelRunner.capture_model`` only drives capture for the
        target's verify-step shapes (multiples of ``1 + K``); it never
        captures at the drafter's native ``num_tokens = bs * K`` shape.
        Without this, runtime FULL dispatch on the drafter would miss
        the wrapper cache and fall through to non-capture replay.

        Must be invoked from inside the runner's ``graph_capture()``
        context (see the call site in ``GPUModelRunner.capture_model``).
        """
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if self.speculative_config.enforce_eager:
            return

        dispatcher = self.cudagraph_dispatcher
        if (
            dispatcher.cudagraph_mode is None
            or dispatcher.cudagraph_mode == CUDAGraphMode.NONE
        ):
            return

        K = self.num_speculative_tokens

        if not self.draft_attn_groups:
            logger.warning(
                "SelfDraftProposer.capture_drafter_cudagraph: "
                "draft_attn_groups is empty; skipping capture."
            )
            return

        batch_sizes = getattr(self, "_self_draft_full_batch_sizes", None) or [1]
        device = self.input_ids.device
        num_warmups = self.compilation_config.cudagraph_num_of_warmups

        def _capture_one(num_reqs: int) -> bool:
            num_tokens = num_reqs * K

            # All metadata fields read through persistent buffers so
            # the captured graph and the propose-time forward see the
            # same device addresses.
            assert hasattr(self, "_persist_query_start_loc"), (
                "Persistent attn-meta buffers were not allocated; "
                "`initialize_cudagraph_keys` must run first."
            )

            qsl_buf = self._persist_query_start_loc
            qsl_cpu_buf = self._persist_query_start_loc_cpu
            torch.arange(0, num_tokens + 1, K, out=qsl_buf[: num_reqs + 1])
            qsl_cpu_buf[: num_reqs + 1] = torch.arange(
                0, num_tokens + 1, K, dtype=qsl_cpu_buf.dtype
            )
            query_start_loc = qsl_buf[: num_reqs + 1]
            query_start_loc_cpu = qsl_cpu_buf[: num_reqs + 1]

            sl_buf = self._persist_seq_lens
            sl_cpu_buf = self._persist_seq_lens_cpu
            sl_buf[:num_reqs].fill_(K)
            sl_cpu_buf[:num_reqs].fill_(K)
            seq_lens = sl_buf[:num_reqs]
            seq_lens_cpu = sl_cpu_buf[:num_reqs]

            sm_buf = self._persist_slot_mapping
            sm_buf[:num_tokens].zero_()
            slot_mapping = sm_buf[:num_tokens]

            block_table_tensor = self._get_persist_block_table(num_reqs)
            if block_table_tensor is None:
                # Standalone-test fallback; in real runs the runner
                # always has an input_batch by this point.
                num_blocks = max(1, (K + self.block_size - 1) // self.block_size)
                block_table_tensor = torch.zeros(
                    (num_reqs, num_blocks),
                    dtype=torch.int32,
                    device=device,
                )

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                _seq_lens_cpu=seq_lens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=K,
                max_seq_len=K,
                block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
            )

            per_layer_attn_metadata: dict[str, object] = {}
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if not hasattr(builder, "build_for_cudagraph_capture"):
                    logger.warning(
                        "SelfDraftProposer.capture_drafter_cudagraph: "
                        "builder %s lacks build_for_cudagraph_capture; "
                        "skipping FULL capture for num_reqs=%d.",
                        type(builder).__name__,
                        num_reqs,
                    )
                    return False
                attn_metadata = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            def _forward(cudagraph_runtime_mode: CUDAGraphMode) -> None:
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=None,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping={
                        layer_name: slot_mapping
                        for layer_name in per_layer_attn_metadata
                    },
                ):
                    input_ids = self.input_ids[:num_tokens]
                    positions = self._get_positions(num_tokens)
                    kwargs = dict(
                        input_ids=input_ids,
                        positions=positions,
                        inputs_embeds=None,
                    )
                    if self.supports_mm_inputs:
                        # Mirror the real forward path: embed once into
                        # the persistent buffer so capture and replay
                        # share the same input-embed address.
                        self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                            input_ids, multimodal_embeddings=None
                        )
                        kwargs = dict(
                            input_ids=None,
                            positions=positions,
                            inputs_embeds=self.inputs_embeds[:num_tokens],
                        )
                    self.model(**kwargs)

            # Warmup outside the wrapper's capture path.
            for _ in range(num_warmups):
                _forward(CUDAGraphMode.NONE)

            _forward(CUDAGraphMode.FULL)
            return True

        captured: list[int] = []
        for bs in batch_sizes:
            if _capture_one(bs):
                captured.append(bs)

        torch.cuda.synchronize()
        logger.info(
            "SelfDraftProposer captured FULL cudagraphs at "
            "num_reqs=%s (num_tokens = num_reqs * K=%d).",
            captured,
            K,
        )
