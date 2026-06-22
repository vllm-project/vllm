# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    AttentionLayerBase = object


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size


class PCPManager:
    """
    Manager for Prefill Context Parallelism (PCP) metadata and buffers.

    This manager encapsulates all PCP-related buffers and logic so that the
    ModelRunner can access them via `self.pcp_manager`.
    """

    num_reqs: int = 0
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        dcp_world_size: int,
        dcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        vllm_config: VllmConfig,
        use_async_scheduling: bool,
        pin_memory: bool = False,
        use_sparse: bool = False,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_world_rank = pcp_rank
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_rank
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1 + (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.vllm_config = vllm_config
        self.max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_async_scheduling = use_async_scheduling
        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_exit_fa_scatter_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.sample_slot_mapping = torch.full(
            (max_buffer_num_tokens,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        # Reinitialized in initialize_slot_mapping.
        self.pcp_padded_slot_mapping_list: list = []
        self.pcp_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.total_num_sampled_tokens_pcp = 0
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask_cpu_tensor = torch.ones(
            (max_buffer_num_tokens,),
            device="cpu",
            dtype=torch.bool,
        )
        self.num_actual_tokens_pcp_padded = 0
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()
        self.full_indices = list(
            range(
                self.max_num_tokens * self.pcp_world_size * self.dcp_world_size
                + self.pcp_world_size * self.dcp_world_size * self.max_num_reqs
            )
        )
        self.use_sparse = use_sparse
        if self.speculative_config and self.pcp_world_size * self.dcp_world_size > 1:
            self.input_ids_pcp_full = CpuGpuBuffer(
                self.max_num_tokens,
                dtype=torch.int32,
                device=device,
                pin_memory=pin_memory,
            )
            self.query_start_loc_pcp_full = CpuGpuBuffer(
                self.max_num_reqs + 1,
                dtype=torch.int32,
                device=device,
                pin_memory=pin_memory,
            )
            self.positions_pcp_full = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int64,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.positions_pcp_full_np = self.positions_pcp_full.numpy()
        self.query_lens_pcp_full = CpuGpuBuffer(
            self.max_num_reqs, dtype=torch.int32, device=device, pin_memory=pin_memory
        )
        self.pcp_fa_query_idx = torch.zeros(
            self.max_num_tokens + 2 * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.pcp_enter_fa_restore_idx = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_world_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.pcp_use_hybrid_attn = (
            self.vllm_config.model_config.hf_config.model_type
            in (
                "qwen3_next",
                "qwen3_5",
                "qwen3_5_moe",
            )
        )

        self.dcp_mtp_attn_mask = CpuGpuBuffer(
            (
                max_num_reqs,
                self.decode_threshold,
                vllm_config.model_config.max_model_len,
            ),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )

        self.pcp_pads_logits_hybrid_attn = torch.ones(
            self.max_num_reqs, dtype=torch.int32
        ) * (self.pcp_world_size - 1)
        self.pcp_padded_tokens_fla = 0
        self.pcp_padded_tokens_length = 0
        self.num_scheduled_tokens_padded: np.ndarray | None = None
        self.max_num_tokens_across_pcp = 0
        self.pcp_tokens_padded = None
        self.total_num_scheduled_tokens = 0
        self._local_num_scheduled_tokens: np.ndarray | None = None
        self._local_total_num_scheduled_tokens: int | None = None

        # Full pre-PCP token layout used to rebuild draft slot mapping
        # after async scheduling corrects num_computed_tokens.
        self.async_rebuild_req_indices_full = None
        self.async_rebuild_cu_num_tokens_full = None
        self.async_rebuild_num_tokens_full = 0

    def _get_cumsum_and_arange(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(
            cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
        )
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def init_batch_info(
        self,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
    ) -> None:
        self.num_reqs = num_reqs
        is_prefill = num_scheduled_tokens[:num_reqs] > self.decode_threshold
        first_prefill = num_reqs if not any(is_prefill) else is_prefill.argmax()
        self.num_decode_reqs = first_prefill
        self.num_prefill_reqs = num_reqs - self.num_decode_reqs
        self.num_decode_tokens = num_scheduled_tokens[: self.num_decode_reqs].sum()
        self.num_scheduled_tokens_padded = (
            num_scheduled_tokens  # for graph compiling in hybrid_attn
        )

        self.query_lens_pcp_full.cpu[: self.num_reqs] = torch.from_numpy(
            num_scheduled_tokens
        )
        self.query_lens_pcp_full.cpu[self.num_reqs :].fill_(0)
        self.query_lens_pcp_full.copy_to_gpu()

    def cache_local_schedule_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        total_num_scheduled_tokens: int,
    ) -> None:
        # Copy to decouple from mutable batch arrays.
        self._local_num_scheduled_tokens = num_scheduled_tokens[:num_reqs].copy()
        self._local_total_num_scheduled_tokens = int(total_num_scheduled_tokens)

    def get_local_schedule_layout(
        self,
    ) -> tuple[np.ndarray | None, int | None]:
        return self._local_num_scheduled_tokens, self._local_total_num_scheduled_tokens

    def fill_prompt_embeds_for_pcp(
        self,
        req_embeds: torch.Tensor,
        req_positions_np: np.ndarray,
        dst_slice: torch.Tensor,
    ) -> None:
        valid_mask_np = req_positions_np < req_embeds.shape[0]
        if not valid_mask_np.any():
            return

        if valid_mask_np.all():
            torch.index_select(
                req_embeds,
                0,
                torch.from_numpy(req_positions_np.astype(np.int64)),
                out=dst_slice,
            )
            return

        src_positions = torch.from_numpy(
            req_positions_np[valid_mask_np].astype(np.int64)
        )
        dst_positions = torch.from_numpy(np.nonzero(valid_mask_np)[0].astype(np.int64))
        dst_slice.index_copy_(
            0, dst_positions, req_embeds.index_select(0, src_positions)
        )

    def build_local_mm_schedule(
        self,
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray,
        encoder_cache: dict[str, torch.Tensor],
    ) -> tuple[dict[str, list[int]], set[str]]:
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        needed_mm_hashes: set[str] = set()

        req_start_idx = 0
        for req_idx, req_id in enumerate(req_ids):
            if req_idx >= local_num_scheduled_tokens.shape[0]:
                break

            num_sched = int(local_num_scheduled_tokens[req_idx])
            if num_sched <= 0:
                req_start_idx += num_sched
                continue

            req_positions = positions_np[req_start_idx : req_start_idx + num_sched]
            req_state = requests[req_id]
            mm_input_ids = list[int]()

            for mm_input_id, mm_feature in enumerate(req_state.mm_features):
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                end_pos = start_pos + pos_info.length
                mm_hash = mm_feature.identifier

                local_mask = (req_positions >= start_pos) & (req_positions < end_pos)
                if not local_mask.any():
                    continue

                local_indices = np.nonzero(local_mask)[0]
                rel_positions = req_positions[local_indices] - start_pos
                is_embed = pos_info.is_embed
                if is_embed is not None:
                    is_embed_np = is_embed.cpu().numpy()
                    if not is_embed_np[rel_positions].any():
                        continue

                needed_mm_hashes.add(mm_hash)
                if mm_hash not in encoder_cache:
                    mm_input_ids.append(mm_input_id)

            if mm_input_ids:
                scheduled_encoder_inputs[req_id] = mm_input_ids

            req_start_idx += num_sched

        return scheduled_encoder_inputs, needed_mm_hashes

    def gather_mm_embeddings_for_pcp(
        self,
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray,
        shift_computed_tokens: int,
        encoder_cache: dict[str, torch.Tensor],
        is_mm_embed: torch.Tensor,
        model: Any,
        is_multimodal_pruning_enabled: bool,
        uses_mrope: bool,
        warning_once: Callable[..., Any] | None = None,
    ) -> tuple[list[torch.Tensor], bool, bool]:
        mm_embeds = list[torch.Tensor]()
        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_idx, req_id in enumerate(req_ids):
            num_sched = int(local_num_scheduled_tokens[req_idx])
            req_positions = positions_np[req_start_idx : req_start_idx + num_sched]
            if shift_computed_tokens:
                req_positions = req_positions + shift_computed_tokens
            req_state = requests[req_id]
            req_taken_mask = np.zeros(num_sched, dtype=np.bool_)
            mm_embeds_req: list[torch.Tensor] = []
            req_mm_local_indices: list[np.ndarray] = []

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                end_pos = start_pos + pos_info.length
                mm_hash = mm_feature.identifier

                local_mask = (req_positions >= start_pos) & (req_positions < end_pos)
                if not local_mask.any():
                    continue

                local_indices = np.nonzero(local_mask)[0]
                rel_positions = req_positions[local_indices] - start_pos

                is_embed = pos_info.is_embed
                if is_embed is not None:
                    is_embed_np = is_embed.cpu().numpy()
                    keep_mask = is_embed_np[rel_positions]
                    if not keep_mask.any():
                        continue
                    local_indices = local_indices[keep_mask]
                    rel_positions = rel_positions[keep_mask]
                    embed_index_map = np.cumsum(is_embed_np.astype(np.int64)) - 1
                    embed_indices = embed_index_map[rel_positions]
                else:
                    embed_indices = rel_positions

                # OR semantics for overlapping mm features: keep first writer.
                keep_new = ~req_taken_mask[local_indices]
                if not keep_new.any():
                    continue
                local_indices = local_indices[keep_new]
                embed_indices = embed_indices[keep_new]
                req_taken_mask[local_indices] = True

                encoder_output = encoder_cache.get(mm_hash)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
                embed_index_tensor = torch.from_numpy(
                    embed_indices.astype(np.int64)
                ).to(
                    device=encoder_output.device,
                    non_blocking=True,
                )
                mm_embeds_item = torch.index_select(
                    encoder_output, 0, embed_index_tensor
                )
                mm_embeds_req.append(mm_embeds_item)
                req_mm_local_indices.append(local_indices.astype(np.int64, copy=False))
                is_mm_embed[req_start_idx + local_indices] = True

            if is_multimodal_pruning_enabled and uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            # Keep multimodal embedding order aligned with is_mm_embed scanning order.
            # Under PCP, request positions may be non-monotonic; concatenating by
            # feature order can misalign embeddings with boolean mask traversal.
            if len(mm_embeds_req) > 1:
                total_local_idx = sum(x.size for x in req_mm_local_indices)
                total_embed_rows = sum(x.shape[0] for x in mm_embeds_req)
                if total_local_idx == total_embed_rows and total_local_idx > 0:
                    local_idx_cat = np.concatenate(req_mm_local_indices, axis=0)
                    embed_cat = torch.cat(mm_embeds_req, dim=0)
                    order = np.argsort(local_idx_cat, kind="stable")
                    order_t = torch.from_numpy(order.astype(np.int64)).to(
                        device=embed_cat.device,
                        non_blocking=True,
                    )
                    mm_embeds_req = [embed_cat.index_select(0, order_t)]
                elif warning_once is not None:
                    warning_once(
                        "PCP MM reorder skipped due to size mismatch: "
                        "local_idx=%d, embed_rows=%d",
                        total_local_idx,
                        total_embed_rows,
                    )

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_sched

        return mm_embeds, should_sync_mrope_positions, should_sync_xdrope_positions

    def maybe_localize_scheduler_output_for_mm_preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray | None,
        local_total_num_scheduled_tokens: int | None,
        encoder_cache: dict[str, torch.Tensor],
    ) -> dict[str, Any] | None:
        need_localize = (
            local_total_num_scheduled_tokens is not None
            and local_total_num_scheduled_tokens
            != scheduler_output.total_num_scheduled_tokens
        )
        if not need_localize and local_num_scheduled_tokens is not None:
            for req_idx, req_id in enumerate(req_ids):
                if req_idx >= local_num_scheduled_tokens.shape[0]:
                    break
                global_sched = scheduler_output.num_scheduled_tokens.get(req_id)
                if global_sched is None or int(global_sched) != int(
                    local_num_scheduled_tokens[req_idx]
                ):
                    need_localize = True
                    break

        if not need_localize:
            return None

        restore_state: dict[str, Any] = {
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "num_scheduled_tokens": scheduler_output.num_scheduled_tokens,
            "scheduled_encoder_inputs": scheduler_output.scheduled_encoder_inputs,
            "free_encoder_mm_hashes": scheduler_output.free_encoder_mm_hashes,
        }

        if local_total_num_scheduled_tokens is not None:
            scheduler_output.total_num_scheduled_tokens = (
                local_total_num_scheduled_tokens
            )

        if local_num_scheduled_tokens is None:
            return restore_state

        num_sched_by_req = dict(scheduler_output.num_scheduled_tokens)
        for req_idx, req_id in enumerate(req_ids):
            if req_idx >= local_num_scheduled_tokens.shape[0]:
                break
            num_sched_by_req[req_id] = int(local_num_scheduled_tokens[req_idx])
        scheduler_output.num_scheduled_tokens = num_sched_by_req

        (
            scheduler_output.scheduled_encoder_inputs,
            local_needed_mm_hashes,
        ) = self.build_local_mm_schedule(
            req_ids=req_ids,
            requests=requests,
            positions_np=positions_np,
            local_num_scheduled_tokens=local_num_scheduled_tokens,
            encoder_cache=encoder_cache,
        )

        # Under PCP, global free list can be earlier than local consumption.
        # Keep MM hashes for all active requests.
        active_mm_hashes = {
            mm_feature.identifier
            for req_state in requests.values()
            for mm_feature in req_state.mm_features
        }
        keep_hashes = active_mm_hashes | local_needed_mm_hashes
        scheduler_output.free_encoder_mm_hashes = [
            mm_hash
            for mm_hash in scheduler_output.free_encoder_mm_hashes
            if mm_hash not in keep_hashes
        ]

        return restore_state

    def restore_scheduler_output_after_mm_preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        restore_state: dict[str, Any] | None,
    ) -> None:
        if restore_state is None:
            return

        scheduler_output.total_num_scheduled_tokens = restore_state[
            "total_num_scheduled_tokens"
        ]
        scheduler_output.num_scheduled_tokens = restore_state["num_scheduled_tokens"]
        scheduler_output.scheduled_encoder_inputs = restore_state[
            "scheduled_encoder_inputs"
        ]
        scheduler_output.free_encoder_mm_hashes = restore_state[
            "free_encoder_mm_hashes"
        ]

    def initialize_slot_mapping(self) -> None:
        """
        Hyrbid-attention models, such as qwen3_next, have plural
        kv_cache_groups, which may lead to problems like overwriting last
        group's pcp_padded_slot_mapping, since they share the same address.
        Therefore we need as many pcp_padded_slot_mappings as kv_cache_groups.
        """
        pcp_padded_slot_mapping = torch.full(
            (self.sample_slot_mapping.shape[0],),
            fill_value=-1,
            dtype=torch.int64,
            device=self.sample_slot_mapping.device,
        )
        self.pcp_padded_slot_mapping_list.append(pcp_padded_slot_mapping)

    def update_tokens_for_pcp(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update token counts and positions for Prefill Context Parallelism (PCP).

        When using Prefill Context Parallelism, each request's prefill sequence is
        split across multiple PCP ranks. The splitting strategy used here is the
        "DualChunkSwap" style: each request's (padded) sequence is split into
        2 * pcp_world_size chunks and ranks are assigned chunks in an interleaved
        head/tail pattern to balance load.

        This function:
        - Computes how many tokens each request should be processed by the current
          PCP rank (pcp_tokens).
        - Computes the flattened positions of those tokens within the local
          padded buffer (pcp_positions).
        - Updates runner state arrays used to restore original order and mask out
          padded tokens after allgather:
            - self.num_pcp_pads_cpu: number of pads added per request
            - self.pcp_unpad_mask_cpu: boolean mask marking real tokens in the
              padded allgather buffer
            - self.pcp_allgather_restore_idx: index array used to restore original
              ordering after per-rank allgather and interleaving.

        Args:
            num_scheduled_tokens: 1D numpy array of length num_reqs containing
                                  the number of new tokens scheduled per request.
            arange_np: 1D numpy array of length max_buffer_num_tokens used for
                       efficient batched arange operations.

        Returns:
            Tuple (pcp_tokens, pcp_positions):
            - pcp_tokens: number of tokens per request that this PCP rank will
                          actually process (after splitting / replication).
                          For hybrid-attention model: number of unpadded tokens
                          per requests
            - pcp_positions: flattened positions for those tokens on this rank,
                             used to build the positions buffer for the model.
        Example:
        >>> Assume tokens = [1, 5, 8], pcp_world_size = 2. After _update_tokens_for_pcp.
        >>> pcp_rank = 0 get ([1, 4, 4], [0, 0, 1, 6, 7, 0, 1, 6, 7])
        >>> pcp_rank = 1 get ([1, 4, 4], [0, 2, 3, 4, 5, 2, 3, 4, 5])
        >>> Meanwhile, the following results are same for each pcp rank
        >>> self.num_pcp_pads_cpu
        [1, 3, 0]
        >>> self.pcp_unpad_mask_cpu
        [True, False, True, True, True, True, True, False, False,
        False, True, True, True, True, True, True, True, True]
        >>> self.pcp_allgather_restore_idx
        [0, 9, 1, 2, 10, 11, 12, 13, 3, 4, 5, 6, 14, 15, 16, 17, 7, 8]
        """

        # DualChunkSwap requires alignment to a multiple of (2 * pcp_world_size).
        # We first pad each request's token count up to that multiple.
        num_padded_scheduled_tokens = np.ceil(
            num_scheduled_tokens / (2 * self.pcp_world_size)
        ).astype(np.int32) * (2 * self.pcp_world_size)

        # PCP does not split decode requests. For decode requests, we instead
        # duplicate the scheduled tokens across the pcp_world_size ranks.
        num_padded_scheduled_tokens[: self.num_decode_reqs] = (
            num_scheduled_tokens[: self.num_decode_reqs] * self.pcp_world_size
        )

        # Record how many pads were added per request (padded - original).
        self.num_pcp_pads_cpu[: self.num_reqs] = (
            num_padded_scheduled_tokens - num_scheduled_tokens
        )

        # cu_padded_tokens: cumulative sum of padded token counts,
        # pcp_padded_arange: per-request arange flattened for padded tokens.
        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )
        self.pcp_padded_tokens_length = pcp_padded_arange.shape[0]
        # Build the mask that marks which positions in the padded allgather buffer
        # correspond to real (unpadded) tokens.
        self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length] = (
            pcp_padded_arange
            < np.repeat(num_scheduled_tokens, num_padded_scheduled_tokens)
        )
        unpad_mask_decode = self.pcp_unpad_mask_cpu[
            : self.num_decode_tokens * self.pcp_world_size
        ]
        unpad_mask_decode = unpad_mask_decode.reshape([-1, self.pcp_world_size])
        unpad_mask_decode[:, 0] = True
        unpad_mask_decode[:, 1:] = False
        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size

        # Compute per-request "chunk sizes" for the head/tail splitting.
        # For prefill requests, we further split the pcp_tokens into two chunks
        # (head and tail). For decode requests, the chunk equals pcp_tokens.
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[: self.num_decode_reqs] = pcp_tokens[: self.num_decode_reqs]

        # Build arange-style helpers for pcp tokens and chunk sizes:
        # - pcp_arange gives indices repeated for each token in pcp_tokens
        # - pcp_chunk_arange gives indices repeated for each position inside chunks
        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes, arange_np)

        # Mask that marks whether a position belongs to the head chunk (True)
        # or the tail chunk (False). For decode requests, tail chunk won't exist
        # and is handled specially below.
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(
            positions_start_loc: int | np.ndarray, rank: int
        ):
            """
            Compute flattened positions for the given rank with a given start
            offset for each request (positions_start_loc).

            - For head chunks: start at positions_start_loc + rank * chunk_size.
            - For tail chunks: start at positions_start_loc + (2*pcp_world_size- rank -
            1) * chunk_size.
            - For decode requests: no tail chunks; their positions are filled from the
              contiguous (unpadded) `tokens` arange instead (handled after).
            """
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = (
                positions_start_loc
                + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            )
            # Fill head positions using chunk arange offset by head_start_loc.
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
                head_start_loc, pcp_chunk_sizes
            )
            # Fill tail positions. Note decode requests do not have tail chunks,
            # so the tail filling is only for prefill positions.
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[self.num_decode_tokens :]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[self.num_decode_tokens :]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_world_rank)
        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0

        # Decode tokens are duplicated only after AG. But their positions are
        # same without prefill context parallel.
        if self.num_decode_reqs > 0:
            positions[: self.num_decode_tokens] = self._get_cumsum_and_arange(
                num_scheduled_tokens[: self.num_decode_reqs], arange_np
            )[1]

        # Build the restore index used after allgather.
        all_positions_lst = [
            get_current_rank_positions(padded_pos_start_loc, rank_i)
            for rank_i in range(self.pcp_world_size)
        ]
        all_positions = np.concatenate(all_positions_lst)
        self.pcp_allgather_restore_idx.np[: all_positions.shape[0]] = (
            all_positions.argsort()
        )
        self.pcp_allgather_restore_idx.copy_to_gpu(all_positions.shape[0])

        self.pcp_tokens[: self.num_reqs] = pcp_tokens[: self.num_reqs]
        self.total_num_sampled_tokens_pcp = pcp_tokens[: self.num_reqs].sum()

        if self.pcp_use_hybrid_attn:
            max_scheduled_prefill_tokens = 0
            self.pcp_padded_tokens_fla = 0
            if self.num_decode_reqs > 0:
                num_padded_scheduled_tokens[: self.num_decode_reqs] = (
                    num_padded_scheduled_tokens[: self.num_decode_reqs]
                    // self.pcp_world_size
                )
            self.total_pcp_padding_tokens_fla = 0
            # have prefills
            if self.num_reqs - self.num_decode_reqs > 0:
                prefill_tokens_tensor = torch.Tensor(
                    num_scheduled_tokens[self.num_decode_tokens :]
                )
                # [num_prefill_reqs, pcp_world_size, 1] [[3,2]] [[2,2,2,1],[2,1,1,1]]
                num_prefill_tokens_allranks = (
                    self._get_cp_local_seq_lens(
                        prefill_tokens_tensor, self.pcp_world_size, 1, 1
                    )
                    .long()
                    .numpy()
                )
                # [3] [2]  |  [2,2] [2,1] [2,1] [1,1]
                num_prefill_scheduled_tokens_linear = num_prefill_tokens_allranks[
                    :, self.pcp_world_rank, 0
                ]
                num_padded_scheduled_tokens[self.num_decode_reqs :] = (
                    num_prefill_scheduled_tokens_linear
                )
                # [[3,5]] | [[0,0,0,0,0],[0,0,0,0,0]]
                num_prefill_tokens_start_loc = np.zeros(
                    (self.num_reqs - self.num_decode_reqs, self.pcp_world_size + 1),
                    dtype=np.int64,
                )
                # [[0,3,5]] | [[0,2,4,6,7],[0,2,3,4,5]]
                num_prefill_tokens_start_loc[:, 1:] = np.cumsum(
                    num_prefill_tokens_allranks[..., 0], axis=-1
                )
                # [0] [3] | [0,0] [2,2] [4,3] [6,4] [7,5]
                num_prefill_tokens_cu_ranks = num_prefill_tokens_start_loc[
                    :, self.pcp_world_rank
                ]
                # [0,1,2] [0,1] | [0,1,0,1] [0,1,0] [0,1,0] [0,0]
                # -> [0,1,2] [3,4] | [0,1,0,1] [2,3,2] [4,5,3] [6,4]
                _, positions_linear = self._get_cumsum_and_arange(
                    num_padded_scheduled_tokens, arange_np
                )
                positions_linear[self.num_decode_reqs :] = positions_linear[
                    self.num_decode_reqs :
                ] + np.repeat(
                    num_prefill_tokens_cu_ranks, num_prefill_scheduled_tokens_linear
                )

                max_scheduled_prefill_tokens = num_prefill_tokens_allranks[
                    :, 0, 0
                ].sum()
                num_prefill_tokens = num_scheduled_tokens[self.num_decode_reqs :].sum()
                self.total_pcp_padding_tokens_fla = (
                    max_scheduled_prefill_tokens * self.pcp_world_size
                    - num_prefill_tokens
                )
                self.pcp_padded_tokens_fla += (
                    max_scheduled_prefill_tokens
                    - num_prefill_scheduled_tokens_linear.sum()
                )

            max_scheduled_tokens = max_scheduled_prefill_tokens + self.num_decode_tokens
            enter_fa_prefill_restore_idx = None
            if self.num_reqs - self.num_decode_reqs > 0:
                # prefill reorder idx
                # [[3,2]] [[2,2,2,1],[2,2,1,1],[1,1,1,1]]
                num_prefill_tokens_allranks = num_prefill_tokens_allranks[..., 0]
                # [0,1,2,0,1] [0,1,0,1,0,1,0,|0,1,0,1,0,0]
                _, prefill_arange_allranks = self._get_cumsum_and_arange(
                    num_prefill_tokens_allranks.flatten(), arange_np
                )
                # [0,1] [0,1,2,3,0,1,2,3]
                _, prefill_rank_offset = self._get_cumsum_and_arange(
                    np.ones(self.num_reqs - self.num_decode_reqs, dtype=np.int64)
                    * self.pcp_world_size,
                    arange_np,
                )
                # [0,0,0,3,3] [0,M,2M,3M,0,M,2M,3M]
                # -> [0,0,M,M,2M,2M,3M,0,0,M,M,2M,3M] + D
                prefill_all_offset = (
                    np.repeat(
                        prefill_rank_offset * max_scheduled_tokens,
                        num_prefill_tokens_allranks.flatten(),
                    )
                    + self.num_decode_tokens
                )

                # [0,0,0,0,|2,2,2,1,|4,4,3,2] -> [0,0,0,0,0,0,0,|2,2,2,2,2,1,|4,4,3,2]
                # [[0,0]] -> [0,0,0,0,0]
                prefill_local_start_local = np.zeros_like(num_prefill_tokens_allranks)
                prefill_local_start_local[1:, :] = np.cumsum(
                    num_prefill_tokens_allranks, axis=0
                )[:-1, :]
                prefill_local_offset = np.repeat(
                    prefill_local_start_local.flatten(),
                    num_prefill_tokens_allranks.flatten(),
                )
                prefill_all_offset = np.add(prefill_all_offset, prefill_local_offset)
                # [0,1,2,3,4]  [0,1,M,M+1,2M,2M+1,3M,0,1,M,M+1,2M,3M]
                enter_fa_prefill_restore_idx = np.add(
                    prefill_all_offset, prefill_arange_allranks
                )
            else:
                _, positions_linear = self._get_cumsum_and_arange(
                    num_padded_scheduled_tokens, arange_np
                )

            # decode reorder idx
            enter_fa_decode_restore_idx = None
            if self.num_decode_reqs > 0:
                # [0,1,2], [4,4,4] -> [0,0,0,0,1,1,1,1,2,2,2,2]
                num_decode_pcp_size = (
                    np.ones(self.num_decode_reqs, dtype=np.int64) * self.pcp_world_size
                )
                decode_reqs_offset = np.repeat(
                    np.arange(self.num_decode_reqs, dtype=np.int64), num_decode_pcp_size
                )
                decode_ranks_offset = (
                    self._get_cumsum_and_arange(num_decode_pcp_size, arange_np)[1]
                    * max_scheduled_tokens
                )
                enter_fa_decode_restore_idx = np.add(
                    decode_reqs_offset, decode_ranks_offset
                )

            if (
                enter_fa_decode_restore_idx is not None
                and enter_fa_prefill_restore_idx is not None
            ):
                pcp_enter_fa_restore_idx = torch.from_numpy(
                    np.concatenate(
                        [enter_fa_decode_restore_idx, enter_fa_prefill_restore_idx]
                    )
                )
            elif enter_fa_decode_restore_idx is not None:
                pcp_enter_fa_restore_idx = torch.from_numpy(enter_fa_decode_restore_idx)

            elif enter_fa_prefill_restore_idx is not None:
                pcp_enter_fa_restore_idx = torch.from_numpy(
                    enter_fa_prefill_restore_idx
                )
            self.pcp_enter_fa_restore_idx[: pcp_enter_fa_restore_idx.shape[0]].copy_(
                pcp_enter_fa_restore_idx.long(), non_blocking=True
            )

            if self.num_reqs > self.num_decode_reqs:
                all_positions_prefill = [
                    get_current_rank_positions(padded_pos_start_loc, rank_i)[
                        self.num_decode_tokens :
                    ]
                    - self.num_decode_tokens * self.pcp_world_size
                    for rank_i in range(self.pcp_world_size)
                ]
                all_positions_prefill_tensor = torch.from_numpy(
                    np.concatenate(all_positions_prefill)
                )
                all_exit_fa_restore_idx = all_positions_prefill_tensor.float().argsort()
                unpad_mask_prefill = self.pcp_unpad_mask_cpu[
                    : self.pcp_padded_tokens_length
                ][self.num_decode_reqs * self.pcp_world_size :]
                # [0] | [0,7]
                ori_tokens_start_loc = np.roll(
                    np.cumsum(num_scheduled_tokens[self.num_decode_tokens :]), 1
                )
                ori_tokens_start_loc[0] = 0
                # [0,1,2] [3,4] | [0,1,7,8] [2,3,9] [4,5,10] [6,11]
                exit_fa_scatter_indices = positions_linear[
                    self.num_decode_reqs :
                ] + np.repeat(ori_tokens_start_loc, num_prefill_scheduled_tokens_linear)

                exit_fa_scatter_idx = torch.index_select(
                    all_exit_fa_restore_idx[unpad_mask_prefill],
                    0,
                    torch.from_numpy(exit_fa_scatter_indices),
                )
                self.pcp_exit_fa_scatter_idx.gpu[: exit_fa_scatter_idx.shape[0]].copy_(
                    exit_fa_scatter_idx.long(), non_blocking=True
                )

                positions_prefill = all_positions_prefill[self.pcp_world_rank]
                pcp_fa_query_idx_tensor = torch.from_numpy(positions_prefill)
                self.pcp_fa_query_idx[: pcp_fa_query_idx_tensor.shape[0]].copy_(
                    pcp_fa_query_idx_tensor.long(), non_blocking=True
                )
            self.pcp_tokens[: self.num_reqs] = pcp_tokens[: self.num_reqs]
            self.total_num_sampled_tokens_pcp = num_scheduled_tokens[
                : self.num_reqs
            ].sum()
            self.max_num_tokens_across_pcp = max_scheduled_tokens
            self.pcp_tokens_padded = pcp_tokens[: self.num_reqs]
            self.num_scheduled_tokens_padded = np.array(
                self.pcp_tokens_padded, dtype=np.int32
            )
            self.total_num_scheduled_tokens = num_padded_scheduled_tokens[
                : self.num_reqs
            ].sum()
            return num_padded_scheduled_tokens, positions_linear
        return pcp_tokens[: self.num_reqs], positions

    def get_logits_indices(
        self,
        cu_num_tokens: np.ndarray,
        num_reqs: int,
        tokens_original: list[int] | None = None,
    ):
        if not self.pcp_use_hybrid_attn or tokens_original is None:
            logits_indices = (
                torch.from_numpy(cu_num_tokens) * self.pcp_world_size
                - self.num_pcp_pads_cpu_tensor[: self.num_reqs]
                - 1
            )
        else:
            tokens_original_tensor = torch.tensor(tokens_original, dtype=torch.int32)
            num_prefill_reqs = (
                (tokens_original_tensor > self.decode_threshold).sum().item()
            )
            num_decode_reqs = num_reqs - num_prefill_reqs
            decode_pads = self.pcp_pads_logits_hybrid_attn[:num_decode_reqs]
            pad_len = tokens_original_tensor.shape[0] - num_decode_reqs
            tokens_logits = tokens_original_tensor + F.pad(
                decode_pads, (0, pad_len), value=0
            )
            logits_indices = torch.cumsum(tokens_logits, dim=0) - 1
        return logits_indices

    def get_padded_slot_mapping(
        self,
        num_tokens: int,
        num_tokens_padded: int,
        slot_mapping: torch.Tensor,
        kv_cache_group_id: int,
    ):
        # After pcp allgather and restore, there are padded tokens in kv,
        # so we need pad slotmapping for alignment.
        while kv_cache_group_id >= len(self.pcp_padded_slot_mapping_list):
            self.initialize_slot_mapping()
        pcp_padded_slot_mapping = self.pcp_padded_slot_mapping_list[kv_cache_group_id]
        if self.pcp_use_hybrid_attn:
            assert self.num_scheduled_tokens_padded is not None
            num_tokens = self.num_scheduled_tokens_padded.sum()
        if (
            not self.pcp_use_hybrid_attn
            or self.total_num_sampled_tokens_pcp != num_tokens_padded
        ):
            pcp_padded_slot_mapping = pcp_padded_slot_mapping[
                : num_tokens_padded * self.pcp_world_size
            ]
        else:
            pcp_padded_slot_mapping = pcp_padded_slot_mapping[
                : num_tokens * self.pcp_world_size
            ]
        cp_unpad_mask = self.pcp_unpad_mask_cpu_tensor[
            : num_tokens * self.pcp_world_size
        ].to(pcp_padded_slot_mapping.device, non_blocking=True)
        num_unpadded_slots = int(cp_unpad_mask.sum().item())
        assert slot_mapping.shape[0] >= num_unpadded_slots, (
            f"PCP slot_mapping has {slot_mapping.shape[0]} entries, but "
            f"{num_unpadded_slots} unpadded slots are required."
        )
        pcp_padded_slot_mapping.fill_(-1)
        pcp_padded_slot_mapping[: num_tokens * self.pcp_world_size][cp_unpad_mask] = (
            slot_mapping[:num_unpadded_slots]
        )
        return pcp_padded_slot_mapping

    def get_restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ):
        # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
        # ignores the padding from CUDA Graph.
        from vllm.distributed.parallel_state import get_pcp_group

        if not self.pcp_use_hybrid_attn:
            hidden_states = get_pcp_group().all_gather(
                hidden_states[
                    : self.num_actual_tokens_pcp_padded // self.pcp_world_size
                ],
                0,
            )
            restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
            return torch.index_select(
                hidden_states,
                0,
                restore_idx,
            )
        else:
            if (
                hidden_states.shape[0] == self.total_num_scheduled_tokens
                and self.pcp_padded_tokens_fla > 0
            ):
                hidden_states = F.pad(
                    hidden_states,
                    pad=(0, 0, 0, self.pcp_padded_tokens_fla),
                    mode="constant",
                    value=0,
                )
            hidden_states = get_pcp_group().all_gather(
                hidden_states[: self.max_num_tokens_across_pcp].contiguous(), dim=0
            )
            restore_idx = self.pcp_enter_fa_restore_idx[
                : hidden_states.shape[0] - self.total_pcp_padding_tokens_fla
            ]
            return torch.index_select(hidden_states, 0, restore_idx)

    def generate_pcp_mtp_input(
        self,
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: dict[str, int],
        with_prefill: bool = True,
        input_batch=None,
        arange_np=None,
        req_indices=None,
        positions_np=None,
        cu_num_tokens=None,
        draft_token_ids=None,
        scheduler_output=None,
        num_spec_tokens=None,
        precomputed_positions_np=None,
    ):
        """
        While pcp > 1, model inputs (input_ids, position, etc.) are split
        across pcp group,
        but mtp need to shift original input_ids before pcp splitting,
        so we record original input_ids here.
        """
        total_num_scheduled_tokens_pcp_full = total_num_scheduled_tokens
        num_scheduled_tokens_pcp_full = np.empty(self.num_reqs, dtype=np.int32)
        for i, req_id in enumerate(input_batch.req_ids):
            num_scheduled_tokens_pcp_full[i] = num_scheduled_tokens[req_id]
        req_indices_pcp_full = np.repeat(
            arange_np[: self.num_reqs], num_scheduled_tokens_pcp_full
        )
        cu_num_tokens_pcp_full = np.cumsum(num_scheduled_tokens_pcp_full)
        self.query_start_loc_pcp_full.np[0] = 0
        self.query_start_loc_pcp_full.np[1 : self.num_reqs + 1] = cu_num_tokens_pcp_full
        self.query_start_loc_pcp_full.np[self.num_reqs + 1 :].fill(-1)
        cumsums_offsets_pcp_full = np.repeat(
            cu_num_tokens_pcp_full - num_scheduled_tokens_pcp_full,
            num_scheduled_tokens_pcp_full,
        )
        arange_pcp_full = (
            arange_np[:total_num_scheduled_tokens_pcp_full] - cumsums_offsets_pcp_full
        )
        positions_pcp_full_np = self.positions_pcp_full_np[
            :total_num_scheduled_tokens_pcp_full
        ]
        if precomputed_positions_np is None:
            np.add(
                input_batch.num_computed_tokens_cpu[req_indices_pcp_full],
                arange_pcp_full,
                out=positions_pcp_full_np,
            )
        else:
            np.copyto(
                positions_pcp_full_np,
                precomputed_positions_np[:total_num_scheduled_tokens_pcp_full],
            )
        token_indices_pcp_full = (
            positions_pcp_full_np
            + req_indices_pcp_full * input_batch.token_ids_cpu.shape[1]
        )
        torch.index_select(
            input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices_pcp_full),
            out=self.input_ids_pcp_full.cpu[:total_num_scheduled_tokens_pcp_full],
        )
        if self.use_async_scheduling:
            self._update_input_ids_pcp_full_ids(
                input_batch,
                draft_token_ids,
                scheduler_output,
                total_num_scheduled_tokens,
                cu_num_tokens_pcp_full,
                num_spec_tokens,
            )
        self.query_start_loc_pcp_full.copy_to_gpu()
        self.input_ids_pcp_full.copy_to_gpu(total_num_scheduled_tokens_pcp_full)
        self.cu_num_tokens_pcp_full = cu_num_tokens_pcp_full

        if self.use_async_scheduling and precomputed_positions_np is None:
            # Save full pre-CP layout so async scheduling can rebuild
            # speculative inputs with corrected num_computed_tokens.
            self.async_rebuild_req_indices_full = req_indices.copy()
            self.async_rebuild_cu_num_tokens_full = cu_num_tokens.copy()
            self.async_rebuild_num_tokens_full = total_num_scheduled_tokens

        # For mtpx, pre-allocate mtp slot_mapping here
        if self.decode_threshold > 2 and not with_prefill:
            num_tokens_ori = sum(list(num_scheduled_tokens.values()))
            num_tokens_mtp = num_tokens_ori + self.num_reqs * (
                self.decode_threshold - 2
            )
            num_tokens_mtp_pad = num_tokens_mtp * self.pcp_world_size
            req_indices_split = np.array_split(req_indices, cu_num_tokens)[
                : self.num_reqs
            ]
            positions_split = np.array_split(positions_np, cu_num_tokens)[
                : self.num_reqs
            ]
            for req_idx in range(self.num_reqs):
                ori_req_indice = req_indices_split[req_idx]
                ori_position = positions_split[req_idx]
                req_indices_split[req_idx] = np.append(
                    ori_req_indice,
                    np.repeat(ori_req_indice[-1], self.decode_threshold - 2),
                )
                positions_split[req_idx] = np.append(
                    ori_position,
                    np.arange(
                        ori_position[-1] + 1,
                        ori_position[-1] + self.decode_threshold - 1,
                    ),
                )
            req_indices_mtp = np.concatenate(req_indices_split)
            positions_mtp = np.concatenate(positions_split)
            input_batch.block_table.compute_slot_mapping_draft(
                req_indices_mtp, positions_mtp
            )
            mtp_slot_ori = input_batch.block_table.block_tables[0].slot_mapping.cpu[
                :num_tokens_mtp
            ]
            unpad_mask = np.repeat(False, num_tokens_mtp_pad)
            unpad_mask[:: self.pcp_world_size] = True
            mtp_slot_pad = torch.full([num_tokens_mtp_pad], -1, dtype=torch.int32)
            mtp_slot_pad[unpad_mask] = mtp_slot_ori
            self.mtp_slot_pad = mtp_slot_pad.to(self.device, non_blocking=True)

    def _update_input_ids_pcp_full_ids(
        self,
        input_batch,
        draft_token_ids,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
        num_spec_tokens: int,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if (
            input_batch.prev_sampled_token_ids is None
            or input_batch.prev_req_id_to_index is None
        ):
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = input_batch.prev_req_id_to_index
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
        num_common_tokens = len(sample_flattened_indices)

        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64
        )
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64
        )
        self.input_ids_pcp_full.cpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ].cpu(),
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64
        )
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64
        )

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = draft_token_ids.to(dtype=torch.int32)

        self.input_ids_pcp_full.cpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor].cpu(),
        )

    def _get_cp_local_seq_lens(
        self,
        seq_lens: torch.Tensor,
        pcp_world_size: int = 1,
        dcp_world_size: int = 1,
        cp_kv_cache_interleave_size: int = 1,
    ) -> torch.Tensor:
        """While using pcp or dcp, kv_cache size stored on each rank may be different,
        use this function to calculate split decode seq_lens of each (p/d)cp rank.
        """
        num_requests = seq_lens.size(0)
        total_world_size = pcp_world_size * dcp_world_size
        seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
        rank_offsets = (
            torch.arange(total_world_size, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(num_requests, 1)
        )
        base = (
            seq_lens_tiled
            // cp_kv_cache_interleave_size
            // total_world_size
            * cp_kv_cache_interleave_size
        )
        remainder = seq_lens_tiled - base * total_world_size
        remainder = torch.clip(
            remainder - rank_offsets * cp_kv_cache_interleave_size,
            0,
            cp_kv_cache_interleave_size,
        )
        dcp_local_seq_lens = (base + remainder).reshape(
            [-1, pcp_world_size, dcp_world_size]
        )
        return dcp_local_seq_lens

    def generate_pcp_metadata(
        self,
        total_num_scheduled_tokens: int,
        query_lens: torch.Tensor,
        input_batch: "InputBatch",
        num_scheduled_tokens: np.ndarray | None,
        block_table_tensor: torch.Tensor,
        num_reqs_padded: int,
        num_reqs: int,
        fixed_decode_seq_lens_cpu: np.ndarray | None = None,
    ):
        from vllm.v1.attention.backend import PrefillContextParallelMetadata

        if self.pcp_world_size > 1 and self.pcp_use_hybrid_attn:
            assert self.num_scheduled_tokens_padded is not None
            total_num_scheduled_tokens = self.num_scheduled_tokens_padded.sum()
        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_world_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        ori_query_lens_cpu = self.query_lens_pcp_full.cpu[:num_reqs_padded]
        if self.pcp_world_size * self.dcp_world_size > 1:
            assert num_scheduled_tokens is not None
            if fixed_decode_seq_lens_cpu is not None:
                decode_context_lens = fixed_decode_seq_lens_cpu[: self.num_decode_reqs]
            else:
                decode_context_lens = (
                    input_batch.num_computed_tokens_cpu[: self.num_decode_reqs]
                    + num_scheduled_tokens[: self.num_decode_reqs]
                )
            prefill_context_lens = input_batch.num_computed_tokens_cpu[
                self.num_decode_reqs : self.num_reqs
            ]
            context_lens = np.concatenate([decode_context_lens, prefill_context_lens])

            num_computed_tokens_of_pcp_dcp = self._get_cp_local_seq_lens(
                torch.tensor(context_lens),
                self.pcp_world_size,
                self.dcp_world_size,
                self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )

            pcp_unpad_mask = self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length]
            long_seq_metadata = PrefillContextParallelMetadata(
                pcp_use_hybrid_attn=self.pcp_use_hybrid_attn,
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.numpy(),
                pcp_unpad_mask=torch.from_numpy(pcp_unpad_mask),
                pcp_padded_tokens_fla=self.pcp_padded_tokens_fla,
                query_lens_pcp_full_cpu=ori_query_lens_cpu,
                max_query_len_pcp_full=ori_query_lens_cpu.max().item(),
            )
            if self.pcp_world_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                kv_tail_proj_idx: list[int] = []
                kv_with_q_head_attn_idx_in_tail, kv_with_q_tail_attn_idx_in_tail = (
                    [],
                    [],
                )
                split_with_q_head_nomask_idx_reqs = []
                split_kv_with_q_tail_nomask_idx_reqs = []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                head_actual_seq_lengths_kv, tail_actual_seq_lengths_kv = [], []
                q_req_offset = 0
                kv_req_offset = 0
                q_head_chunk_id = self.pcp_world_rank
                q_tail_chunk_id = self.pcp_world_size * 2 - 1 - self.pcp_world_rank
                for i, seq_len in enumerate(query_lens):
                    if i < self.num_decode_reqs:
                        continue
                    chunk_len = seq_len // 2
                    chunk_seqlens.append(chunk_len)
                    q_head_idx.extend(
                        list(range(q_req_offset, q_req_offset + chunk_len))
                    )
                    kv_with_q_head_nomask_idx.extend(
                        list(
                            range(
                                kv_req_offset,
                                kv_req_offset + chunk_len * q_head_chunk_id,
                            )
                        )
                    )
                    kv_with_q_head_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_head_chunk_id,
                                kv_req_offset + chunk_len * (q_head_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)
                    split_with_q_head_nomask_idx_reqs.append(
                        list(
                            range(
                                kv_req_offset,
                                kv_req_offset + chunk_len * q_head_chunk_id,
                            )
                        )
                    )
                    q_tail_idx.extend(
                        list(
                            range(
                                q_req_offset + chunk_len, q_req_offset + chunk_len * 2
                            )
                        )
                    )
                    kv_with_q_tail_nomask_idx.extend(
                        list(
                            range(
                                kv_req_offset,
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                            )
                        )
                    )
                    kv_with_q_tail_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                                kv_req_offset + chunk_len * (q_tail_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)
                    split_kv_with_q_tail_nomask_idx_reqs.append(
                        list(
                            range(
                                kv_req_offset,
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                            )
                        )
                    )
                    tail_proj_offset = len(kv_tail_proj_idx)
                    tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
                    kv_tail_proj_idx.extend(
                        list(range(kv_req_offset, kv_req_offset + tail_proj_len))
                    )
                    kv_with_q_head_attn_idx_in_tail.extend(
                        list(
                            range(
                                tail_proj_offset,
                                tail_proj_offset + chunk_len * (q_head_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_tail_attn_idx_in_tail.extend(
                        list(range(tail_proj_offset, tail_proj_offset + tail_proj_len))
                    )
                    head_actual_seq_lengths_kv.append(
                        len(kv_with_q_head_attn_idx_in_tail)
                    )
                    tail_actual_seq_lengths_kv.append(
                        len(kv_with_q_tail_attn_idx_in_tail)
                    )
                    q_req_offset += seq_len
                    kv_req_offset += seq_len * self.pcp_world_size

                q_head_idx_tensor = self._list_to_tensor(q_head_idx, self.device)
                q_tail_idx_tensor = self._list_to_tensor(q_tail_idx, self.device)
                self.q_head_idx_tensor = q_head_idx_tensor
                self.q_tail_idx_tensor = q_tail_idx_tensor

                q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
                q_full_idx = q_full_idx.to(torch.float32).argsort().to(torch.int32)
                self.q_full_idx = q_full_idx

                self.kv_idx_names = {
                    "kv_with_q_head_nomask_idx_tensor": kv_with_q_head_nomask_idx,
                    "kv_with_q_head_mask_idx_tensor": kv_with_q_head_mask_idx,
                    "kv_with_q_tail_nomask_idx_tensor": kv_with_q_tail_nomask_idx,
                    "kv_with_q_tail_mask_idx_tensor": kv_with_q_tail_mask_idx,
                    "kv_tail_proj_idx_tensor": kv_tail_proj_idx,
                    "kv_with_q_head_attn_idx_in_tail_tensor": (
                        kv_with_q_head_attn_idx_in_tail
                    ),
                    "kv_with_q_tail_attn_idx_in_tail_tensor": (
                        kv_with_q_tail_attn_idx_in_tail
                    ),
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = self._list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_chunk_seqlens = torch.tensor(chunk_seqlens, dtype=torch.int32)
                attn_mask_seqlens = torch.cumsum(
                    torch.tensor(chunk_seqlens, dtype=torch.int32), dim=0
                ).tolist()
                head_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_head_nomask_seqlens, dtype=torch.int32),
                    dim=0,
                ).tolist()
                tail_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_tail_nomask_seqlens, dtype=torch.int32),
                    dim=0,
                ).tolist()

                self.extra_long_seq_kwargs = {
                    "attn_mask_seqlens": attn_mask_seqlens,
                    "head_attn_nomask_seqlens": head_attn_nomask_seqlens,
                    "tail_attn_nomask_seqlens": tail_attn_nomask_seqlens,
                    "head_actual_seq_lengths_kv": head_actual_seq_lengths_kv,
                    "tail_actual_seq_lengths_kv": tail_actual_seq_lengths_kv,
                }
                long_seq_metadata.pcp_allgather_restore_idx = (
                    self.pcp_allgather_restore_idx.gpu[:num_actual_tokens_pcp_padded]
                )
                if self.pcp_use_hybrid_attn:
                    long_seq_metadata.pcp_exit_fa_scatter_idx = (
                        self.pcp_exit_fa_scatter_idx.gpu[
                            : num_scheduled_tokens.sum() - self.num_decode_reqs
                        ]
                    )
                    long_seq_metadata.pcp_fa_query_idx = self.pcp_fa_query_idx[
                        : num_actual_tokens_pcp_padded // self.pcp_world_size
                        - self.num_decode_reqs
                    ]
                    long_seq_metadata.pcp_enter_fa_restore_idx = (
                        self.pcp_enter_fa_restore_idx[
                            : pcp_unpad_mask.sum()
                            + self.num_decode_reqs * (self.pcp_world_size - 1)
                        ]
                    )
                    long_seq_metadata.max_num_tokens_across_pcp = (
                        self.max_num_tokens_across_pcp
                    )
                    long_seq_metadata.total_num_scheduled_tokens = (
                        self.total_num_scheduled_tokens
                    )
                long_seq_metadata.q_head_idx_tensor = self.q_head_idx_tensor
                long_seq_metadata.q_tail_idx_tensor = self.q_tail_idx_tensor
                long_seq_metadata.q_full_idx = self.q_full_idx
                long_seq_metadata.kv_with_q_head_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_head_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_head_mask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_head_mask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_tail_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_mask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_tail_mask_idx_tensor"
                ]
                long_seq_metadata.kv_tail_proj_idx_tensor = self.kv_idx_names[
                    "kv_tail_proj_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_head_attn_idx_in_tail_tensor = (
                    self.kv_idx_names["kv_with_q_head_attn_idx_in_tail_tensor"]
                )
                long_seq_metadata.kv_with_q_tail_attn_idx_in_tail_tensor = (
                    self.kv_idx_names["kv_with_q_tail_attn_idx_in_tail_tensor"]
                )
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs[
                    "attn_mask_seqlens"
                ]
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    "head_attn_nomask_seqlens"
                ]
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs[
                    "tail_attn_nomask_seqlens"
                ]
                long_seq_metadata.head_actual_seq_lengths_kv = (
                    self.extra_long_seq_kwargs["head_actual_seq_lengths_kv"]
                )
                long_seq_metadata.tail_actual_seq_lengths_kv = (
                    self.extra_long_seq_kwargs["tail_actual_seq_lengths_kv"]
                )
                long_seq_metadata.attn_chunk_seqlens = attn_chunk_seqlens

            # Generate MTP attention masks for decode requests when dcp_size
            # > 1 with speculative decoding.
            if (
                self.dcp_world_size * self.pcp_world_size > 1
                and self.speculative_config
                and self.num_decode_reqs > 0
                and num_scheduled_tokens is not None
            ):
                # Extract decode request info from input_batch and num_scheduled_tokens
                decode_num_scheduled_tokens = num_scheduled_tokens[
                    : self.num_decode_reqs
                ]
                if fixed_decode_seq_lens_cpu is not None:
                    decode_num_computed_tokens = (
                        fixed_decode_seq_lens_cpu[: self.num_decode_reqs]
                        - decode_num_scheduled_tokens
                    ).tolist()
                else:
                    decode_num_computed_tokens = input_batch.num_computed_tokens_cpu[
                        : self.num_decode_reqs
                    ].tolist()

                dcp_mtp_attn_mask = self.generate_mtp_attention_mask_for_decode(
                    decode_num_computed_tokens, decode_num_scheduled_tokens
                )
                if dcp_mtp_attn_mask is not None:
                    self.dcp_mtp_attn_mask.np[: self.num_decode_reqs] = (
                        dcp_mtp_attn_mask
                    )
                    self.dcp_mtp_attn_mask.copy_to_gpu(self.num_decode_reqs)
                    long_seq_metadata.dcp_mtp_attn_mask = self.dcp_mtp_attn_mask.gpu[
                        : self.num_decode_reqs
                    ]
                else:
                    long_seq_metadata.dcp_mtp_attn_mask = None
            else:
                long_seq_metadata.dcp_mtp_attn_mask = None

        self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata, block_table_tensor

    def _list_to_tensor(self, lst, device, dtype=torch.int32):
        tensor_npu = torch.zeros(len(lst), dtype=dtype, device=device)
        tensor_npu.copy_(torch.tensor(lst, dtype=dtype), non_blocking=True)
        return tensor_npu

    def remap_mrope_positions_for_pcp(
        self,
        positions_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        input_batch: "InputBatch",
        requests: dict[str, Any],
        mrope_positions: CpuGpuBuffer,
    ):
        """Remap mrope_positions after PCP split.

        _calc_mrope_positions fills mrope_positions using the original
        (pre-PCP-split) sequential token ordering from scheduler_output.
        After PCP splits tokens across ranks, each rank only processes a
        subset of tokens (head+tail chunks), so we must remap mrope_positions
        to match the PCP-local token ordering.

        positions_np already contains the correct absolute position for each
        token on this PCP rank (computed by update_tokens_for_pcp). We use
        these positions to gather the correct mrope_positions from
        req.mrope_positions (for prompt tokens) or compute them on-the-fly
        (for completion/decode tokens).
        """
        mrope_pos_ptr = 0
        for index, req_id in enumerate(input_batch.req_ids):
            req = requests[req_id]
            num_sched = int(num_scheduled_tokens[index])
            local_positions = positions_np[mrope_pos_ptr : mrope_pos_ptr + num_sched]

            if req.mrope_positions is not None and req.mrope_positions.shape[1] > 0:
                num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                    req.prompt_token_ids, req.prompt_embeds
                )
                max_mrope_idx = req.mrope_positions.shape[1]

                # Build the mrope_positions for this request's PCP-local
                # tokens. For each token, gather from req.mrope_positions
                # using its absolute position from positions_np.
                mrope_dst = np.empty((3, num_sched), dtype=np.int64)

                # Prompt tokens: positions within prompt range,
                # gather from pre-computed req.mrope_positions.
                prompt_mask = local_positions < min(num_prompt_tokens, max_mrope_idx)
                if prompt_mask.any():
                    prompt_indices = local_positions[prompt_mask].astype(np.int64)
                    prompt_indices = np.clip(prompt_indices, 0, max_mrope_idx - 1)
                    mrope_dst[:, prompt_mask] = req.mrope_positions[
                        :, torch.from_numpy(prompt_indices)
                    ].numpy()

                # Completion/decode tokens: all 3 dims use the same
                # position.
                completion_mask = local_positions >= num_prompt_tokens
                if completion_mask.any():
                    # For completion tokens, use mrope_position_delta to
                    # compute the correct position, same as
                    # get_next_input_positions_tensor.
                    if req.mrope_position_delta is not None:
                        comp_positions = (
                            local_positions[completion_mask] + req.mrope_position_delta
                        )
                    else:
                        comp_positions = local_positions[completion_mask]
                    mrope_dst[:, completion_mask] = comp_positions[np.newaxis, :]

                # Padding tokens beyond req.mrope_positions shape:
                # use the last valid mrope position.
                padding_mask = (~prompt_mask) & (~completion_mask)
                if padding_mask.any():
                    last_idx = max_mrope_idx - 1
                    mrope_dst[:, padding_mask] = req.mrope_positions[
                        :, last_idx : last_idx + 1
                    ].numpy()

                mrope_positions.cpu[:, mrope_pos_ptr : mrope_pos_ptr + num_sched] = (
                    torch.from_numpy(mrope_dst)
                )
            else:
                # No mrope_positions available:
                # all 3 dims equal the 1D position.
                mrope_positions.cpu[:, mrope_pos_ptr : mrope_pos_ptr + num_sched] = (
                    torch.from_numpy(local_positions[np.newaxis, :].astype(np.int64))
                )

            mrope_pos_ptr += num_sched

    def generate_mtp_attention_mask_for_decode(
        self,
        decode_num_computed_tokens: list[int],
        decode_num_scheduled_tokens: np.ndarray,
    ) -> list[torch.Tensor | None]:
        """
        Generate MTP attention masks for decode requests in PCP mode.

        This function handles the case where decode requests with MTP
        (speculative decoding) need attention masks computed based on the
        local sequence after load balancing.

        New MTP token allocation logic (using position % cp_size):
        - History tokens are already split via DualChunkSwap
        - MTP tokens are allocated based on (history_len + mtp_idx) % cp_size
        - Each rank only computes mask for tokens assigned to itself

        Example:
            - pcp=1, dcp=2 (cp_size=2)
            - history_len=5: [a,b,c,d,e] split via DualChunkSwap
              - cp0: [a,b,c] (positions 0,1,2) -> 3 tokens
              - cp1: [d,e] (positions 3,4) -> 2 tokens
            - num_scheduled_tokens=4: [f,g,h,i] (positions 5,6,7,8)
            - MTP allocation by position % cp_size:
              - f: pos 5 % 2 = 1 -> rank1
              - g: pos 6 % 2 = 0 -> rank0
              - h: pos 7 % 2 = 1 -> rank1
              - i: pos 8 % 2 = 0 -> rank0
            - Final:
              - rank0: [a,b,c,g,i] positions [0,1,2,6,8] -> mask shape 4x5
              - rank1: [d,e,f,h] positions [3,4,5,7] -> mask shape 4x4

        Args:
            decode_num_computed_tokens: List of global history lengths for
                decode requests
            decode_num_scheduled_tokens: Array of scheduled token counts for
                decode requests
        """
        cp_rank = self.pcp_world_rank * self.dcp_world_size + self.dcp_world_rank
        cp_size = self.pcp_world_size * self.dcp_world_size
        assert cp_size > 1, "cp_size must be greater than 1"

        q_lens = torch.tensor(
            decode_num_scheduled_tokens[: self.num_decode_reqs], dtype=torch.int32
        )
        global_histories = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
        total_lens = global_histories + q_lens
        context_lens = total_lens - q_lens

        max_indices = total_lens - 1
        valid = max_indices >= cp_rank

        if not valid.any():
            return self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]

        k_lens = torch.div(max_indices - cp_rank, cp_size, rounding_mode="floor") + 1
        k_lens = torch.where(valid, k_lens, torch.zeros_like(k_lens))

        mtp_attn_mask = self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]
        mtp_attn_mask.zero_()

        num_valid = valid.sum().item()
        if num_valid == 0:
            return mtp_attn_mask

        max_q = int(q_lens[valid].max().item())
        max_k = int(k_lens[valid].max().item())

        # Generate indices up to max dimensions
        q_indices = torch.arange(max_q, dtype=torch.int32)
        k_indices = torch.arange(max_k, dtype=torch.int32)

        valid_q = valid[:, None] & (q_indices[None, :] < q_lens[:, None])
        valid_k = valid[:, None] & (k_indices[None, :] < k_lens[:, None])

        k_upper = (context_lens[:, None] + q_indices - cp_rank) // cp_size
        k_upper_expanded = k_upper[:, :, None]  # [num_decode_reqs, max_q, 1]
        k_idx_expanded = k_indices[None, None, :]  # [1, 1, max_k]
        full_mask = (k_idx_expanded > k_upper_expanded) & (k_upper_expanded >= 0)

        valid_mask_3d = valid_q[:, :, None] & valid_k[:, None, :]
        full_mask = full_mask & valid_mask_3d

        mtp_attn_mask[: self.num_decode_reqs, :max_q, :max_k] = full_mask

        return mtp_attn_mask
