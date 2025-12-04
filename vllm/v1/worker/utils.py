# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.cache import processor_only_cache_from_config
from vllm.multimodal.registry import MultiModalRegistry
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec
from vllm.v1.utils import CpuGpuBuffer


class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        mm_registry: MultiModalRegistry,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.mm_registry = mm_registry
        self.cache = cache = processor_only_cache_from_config(model_config, mm_registry)

        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = scheduler_config.max_num_seqs

        self.mm_limits = mm_registry.get_mm_limits_per_prompt(model_config, cache=cache)

        max_tokens_by_modality = mm_registry.get_max_tokens_per_item_by_modality(
            model_config,
            cache=cache,
            profiler_limits=self.mm_limits,
        )

        encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        max_items_per_prompt_by_modality = dict[str, int]()
        max_items_per_batch_by_modality = dict[str, int]()

        for modality, max_tokens in max_tokens_by_modality.items():
            (
                max_items_per_prompt,
                max_items_per_batch,
            ) = self.get_max_items(modality, max_tokens)

            max_items_per_prompt_by_modality[modality] = max_items_per_prompt
            max_items_per_batch_by_modality[modality] = max_items_per_batch

        self.max_tokens_by_modality = max_tokens_by_modality
        self.max_items_per_prompt_by_modality = max_items_per_prompt_by_modality
        self.max_items_per_batch_by_modality = max_items_per_batch_by_modality

    def get_modality_with_max_tokens(self) -> str:
        max_tokens_by_modality = self.max_tokens_by_modality
        modality, _ = max(max_tokens_by_modality.items(), key=lambda x: x[1])

        return modality

    def get_encoder_budget(self) -> int:
        return min(self.encoder_compute_budget, self.encoder_cache_size)

    def get_max_items(
        self,
        modality: str,
        max_tokens_per_item: int,
    ) -> tuple[int, int]:
        if max_tokens_per_item == 0:
            return 0, 0

        # Check how many items of this modality can be supported by
        # the encoder budget.
        encoder_budget = self.get_encoder_budget()

        # TODO: handle encoder-decoder models once we support them.
        if encoder_budget == 0:
            return 0, 0

        max_encoder_items_per_batch = encoder_budget // max_tokens_per_item

        # Check how many items of this modality can be supported by
        # the decoder budget.
        mm_limit = self.mm_limits[modality]

        max_items_per_prompt = max(
            1,
            min(mm_limit, self.max_model_len // max_tokens_per_item),
        )

        scheduler_config = self.scheduler_config
        max_num_reqs = self.max_num_reqs

        if not scheduler_config.enable_chunked_prefill:
            max_num_reqs = min(
                max_num_reqs,
                scheduler_config.max_num_batched_tokens // max_tokens_per_item,
            )

        max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt

        max_items_per_batch = max(
            1,
            min(max_encoder_items_per_batch, max_decoder_items_per_batch),
        )

        return max_items_per_prompt, max_items_per_batch

    def reset_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear_cache()


class PCPManager:
    """
    Manager for Prefill Context Parallelism (PCP) metadata and buffers.

    This manager encapsulates all PCP-related buffers and logic so that the
    ModelRunner can access them via `self.pcp_manager`.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank

        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_padded_slot_mapping = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask_cpu_tensor = torch.zeros(
            (max_buffer_num_tokens,),
            device="cpu",
            dtype=torch.bool,
        )
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()

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

    def update_tokens_for_pcp(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        num_reqs: int,
        reorder_batch_threshold: int | None = None,
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
            num_reqs: Total number of requests in the batch.
            reorder_batch_threshold: Threshold for decode vs prefill requests.

        Returns:
            Tuple (pcp_tokens, pcp_positions):
            - pcp_tokens: number of tokens per request that this PCP rank will
                          actually process (after splitting / replication).
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
        >>> self.pcp_allgather_resotre_idx
        [0, 9, 1, 2, 10, 11, 12, 13, 3, 4, 5, 6, 14, 15, 16, 17, 7, 8]
        """

        assert reorder_batch_threshold is not None, (
            "PCP depends on reorder batch to split decode and prefill requests."
        )
        num_decode_reqs = sum(num_scheduled_tokens <= reorder_batch_threshold)
        num_decode_tokens = sum(num_scheduled_tokens[:num_decode_reqs])

        # DualChunkSwap requires alignment to a multiple of (2 * pcp_world_size).
        # We first pad each request's token count up to that multiple.
        num_padded_scheduled_tokens = np.ceil(
            num_scheduled_tokens / (2 * self.pcp_world_size)
        ).astype(np.int32) * (2 * self.pcp_world_size)

        # PCP does not split decode requests. For decode requests, we instead
        # duplicate the scheduled tokens across the pcp_world_size ranks.
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            num_scheduled_tokens[:num_decode_reqs] * self.pcp_world_size
        )

        # Record how many pads were added per request (padded - original).
        self.num_pcp_pads_cpu[:num_reqs] = (
            num_padded_scheduled_tokens - num_scheduled_tokens
        )

        # cu_padded_tokens: cumulative sum of padded token counts,
        # pcp_padded_arange: per-request arange flattened for padded tokens.
        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )
        # Build the mask that marks which positions in the padded allgather buffer
        # correspond to real (unpadded) tokens.
        self.pcp_unpad_mask_cpu[: pcp_padded_arange.shape[0]] = (
            pcp_padded_arange
            < np.repeat(num_scheduled_tokens, num_padded_scheduled_tokens)
        )

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size

        # Compute per-request "chunk sizes" for the head/tail splitting.
        # For prefill requests, we further split the pcp_tokens into two chunks
        # (head and tail). For decode requests, the chunk equals pcp_tokens.
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

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
                pcp_chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_rank)
        # Decode tokens are duplicated only after AG. But their positions are
        # same without prefill context parallel.
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                num_scheduled_tokens[:num_decode_reqs], arange_np
            )[1]

        # Build the restore index used after allgather.
        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions_lst = [
            get_current_rank_positions(padded_pos_start_loc, rank_i)
            for rank_i in range(self.pcp_world_size)
        ]
        all_positions = np.concatenate(all_positions_lst)
        self.pcp_allgather_restore_idx.np[: all_positions.shape[0]] = (
            all_positions.argsort()
        )
        self.pcp_allgather_restore_idx.copy_to_gpu(all_positions.shape[0])

        return (
            pcp_tokens[:num_reqs],
            positions,
        )

    def get_logits_indices(self, cu_num_tokens: np.ndarray, num_reqs: int):
        return (
            torch.from_numpy(cu_num_tokens) * self.pcp_world_size
            - self.num_pcp_pads_cpu_tensor[:num_reqs]
            - 1
        )

    def get_discard_request_mask(
        self,
        num_computed_tokens_cpu: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        num_tokens_np: np.ndarray,
    ):
        return (
            num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens * self.pcp_world_size
            - self.num_pcp_pads_cpu[:num_reqs]
        ) < num_tokens_np

    def get_padded_slot_mapping(self, num_tokens: int, slot_mapping: torch.Tensor):
        # After pcp allgather and restore, there are padded tokens in kv,
        # so we need pad slotmapping for alignment.
        pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[
            : num_tokens * self.pcp_world_size
        ]
        cp_unpad_mask = self.pcp_unpad_mask_cpu_tensor[
            : num_tokens * self.pcp_world_size
        ]
        pcp_padded_slot_mapping.fill_(-1)
        pcp_padded_slot_mapping[cp_unpad_mask] = slot_mapping
        return pcp_padded_slot_mapping

    def get_restore_hidden_states(
        self, hidden_states: torch.Tensor, num_tokens_unpadded: int
    ):
        # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
        # ignores the padding from CUDA Graph.
        hidden_states = get_pcp_group().all_gather(
            hidden_states[:num_tokens_unpadded],
            0,
        )
        restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
        return torch.index_select(
            hidden_states,
            0,
            restore_idx,
        )


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistant buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None,
        num_metadata_builders: int = 1,
    ):
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: torch.Tensor | None,
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
            Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
            tokens need to be filled with multimodal embeddings.
            Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: torch.Tensor | None,
) -> torch.Tensor:
    """
    Reconstructs the embeddings from the placeholder tokens.

    This is the operation of [`scatter_mm_placeholders`]
    [vllm.v1.worker.utils.scatter_mm_placeholders].
    """
    if is_embed is None:
        return placeholders

    return placeholders[is_embed]


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
    """
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                # We know that the GPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                pass
            else:
                raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled.

    This follows the same logic as SequenceParallelismPass.is_applicable_for_range():
    - In full-graph compilation mode (no splitting ops or using inductor graph
      partition), SP is always applied
    - Otherwise, SP is only applied for specific shapes in compile_sizes
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        return False

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    assert num_input_tokens % tp == 0

    if (
        not vllm_config.compilation_config.splitting_ops
        or vllm_config.compilation_config.use_inductor_graph_partition
    ):
        return True
    compile_sizes = vllm_config.compilation_config.compile_sizes
    if compile_sizes is None:
        return False
    return num_input_tokens in compile_sizes
