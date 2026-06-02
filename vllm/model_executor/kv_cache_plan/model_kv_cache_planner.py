# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default model KV cache planner."""

from vllm.utils.math_utils import cdiv

from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.kv_cache_plan.kv_cache_planner import KVCachePlanner
from vllm.model_executor.kv_cache_plan.utils import (
    _get_kv_cache_groups_uniform_page_size,
    _get_kv_cache_groups_uniform_spec,
    _get_kv_cache_groups_uniform_type,
    get_uniform_page_size,
    is_kv_cache_spec_uniform,
    is_kv_cache_type_attention_free,
    unify_hybrid_kv_cache_specs,
    unify_kv_cache_spec_page_size,
)
from vllm.models.deepseek_v4.nvidia.kv_cache_planner_utils import (
    check_enough_kv_cache_memory,
    estimate_max_model_len_from_groups,
    get_num_blocks,
    may_override_num_blocks,
    merge_kv_cache_specs_across_workers,
    pool_bytes_per_block,
    project_kv_cache_groups_to_worker,
    report_kv_cache_config,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


class DefaultModelKVCachePlanner(KVCachePlanner):
    """Default KV cache planner used by models without a custom planner."""

    def get_kv_cache_configs(
        self,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        """
        Generates the KV cache configurations for a model.
        Since we use a shared centralized controller for all workers, we need the
        `kv_cache_config` to be consistent across all workers to make sure
        the KV cache allocation can be applied to all workers. However, different
        workers may have different memory available, and different type of layers
        (when pipeline parallel is enabled). To handle the difference between
        workers, the current implementation is:
        1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
        the whole model.
        2. Generate the KV cache groups based on the layer ratio of the whole model.
        This also handles spec unification for hybrid models.
        3. Handle auto-fit max_model_len and memory checks using per-worker
        projected groups to account for PP sharding.
        4. Generate the KV cache configs for each worker based on the KV cache
        grouping strategy. (This is reasonable because the layer ratio of
        different PP stages are similar.)
        5. Change the num_blocks of each worker to the smallest among all workers
        and shrink tensor sizes proportionally to avoid allocating unused memory.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
            available_memory: Memory available for KV cache in bytes for each
                worker.

        Returns:
            The generated KVCacheConfigs for each worker.
        """

        merged_kv_cache_specs = merge_kv_cache_specs_across_workers(kv_cache_specs)
        # Get global KV cache groups. This also handles spec unification for
        # hybrid models when disable_hybrid_kv_cache_manager is enabled.
        # After this call, merged_kv_cache_specs may be modified in-place.
        global_kv_cache_groups = self.get_kv_cache_groups(merged_kv_cache_specs)

        # If original_max_model_len was -1, automatically
        # determine the maximum model length that fits in available GPU memory.
        # We use per-worker projected groups to account for PP sharding.
        projected_groups_per_worker = [
            project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
            for worker_spec in kv_cache_specs
        ]

        # If `num_gpu_blocks_override` is set, the cache size that will actually
        # be allocated is decoupled from the profiled `available_memory`:
        # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
        # `num_blocks` to the override. Reflect that in `available_memory` here so
        # auto-fit, the admission check, and the per-worker config builder all
        # plan against the same effective capacity.
        override = self.cache_config.num_gpu_blocks_override
        if override is not None:
            adjusted_memory: list[int] = []
            for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
                if not groups:
                    adjusted_memory.append(avail_mem)
                    continue
                bytes_per_block = pool_bytes_per_block(groups)
                logger.info(
                    "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                    avail_mem // bytes_per_block,
                    override,
                )
                adjusted_memory.append(override * bytes_per_block)
            available_memory = adjusted_memory

        if self.vllm_config.model_config.original_max_model_len == -1:
            self._auto_fit_max_model_len(projected_groups_per_worker, available_memory)

        # Check if the available memory is enough per worker.
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                continue
            check_enough_kv_cache_memory(
                avail_mem,
                lambda groups=groups: self._max_memory_usage_bytes_from_groups(
                    groups
                ),
                self.vllm_config.model_config.max_model_len,
                lambda available_memory, groups=groups: (
                    self.get_max_model_len_capacity(groups, available_memory)
                ),
            )

        kv_cache_configs: list[KVCacheConfig] = []
        for (
            projected_groups,
            kv_cache_spec_one_worker,
            available_memory_one_worker,
        ) in zip(
            projected_groups_per_worker,
            kv_cache_specs,
            available_memory,
        ):
            assert sum(len(group.layer_names) for group in projected_groups) == len(
                kv_cache_spec_one_worker
            ), "Some layers are not assigned to any group."
            kv_cache_configs.append(
                self.get_kv_cache_config_from_groups(
                    projected_groups,
                    available_memory_one_worker,
                )
            )

        # Change the num_blocks of each rank to the smallest among all ranks.
        # We also need to shrink the tensor size proportionally to avoid
        # allocating unused memory.
        min_num_blocks = min(
            kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
        )
        for kv_cache_config in kv_cache_configs:
            num_blocks_old = kv_cache_config.num_blocks
            kv_cache_config.num_blocks = min_num_blocks

            # Shrink tensor size proportionally
            for tensor in kv_cache_config.kv_cache_tensors:
                assert tensor.size % num_blocks_old == 0
                tensor.size = tensor.size // num_blocks_old * min_num_blocks

            if len(kv_cache_config.kv_cache_groups) > 0:
                report_kv_cache_config(self.vllm_config, kv_cache_config)

        return kv_cache_configs

    def get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """
        Split the layers in the model into groups with the same KV cache spec.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_spec: The kv cache spec of each attention layer in the model

        Returns:
            The generated KVCacheGroups
        """
        if self.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            unify_hybrid_kv_cache_specs(kv_cache_specs)

        if is_kv_cache_type_attention_free(kv_cache_specs):
            # This returns an empty list to allow for the KVCacheManager to handle
            # attention free models.
            return []

        if is_kv_cache_spec_uniform(kv_cache_specs):
            # KV cache of all layers are the same, which is true for
            # most models. Allocate the same amount of memory for
            # each layer.
            return _get_kv_cache_groups_uniform_spec(kv_cache_specs)
        elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_specs):
            # All layers need the same number of token slots (e.g., all layers are
            # full attention, or all layers are sliding window attention with the
            # same window size). Put all layers into one group.
            return _get_kv_cache_groups_uniform_type(uniform_spec)

        # Pull HiddenStateCacheSpec layers out before the general multi-group
        # path so they don't affect page-size unification or grouping.
        hidden_specs = {
            k: v
            for k, v in kv_cache_specs.items()
            if isinstance(v, HiddenStateCacheSpec)
        }
        filtered_spec = {
            k: v
            for k, v in kv_cache_specs.items()
            if not isinstance(v, HiddenStateCacheSpec)
        }

        # As KVCacheManager can only allocate memory of one size, we need to unify
        # the page size of the layers. For cases cannot be unified, this function
        # will raise an error.
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
        groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

        # Add hidden-state layers back with page aligned to the common page.
        if hidden_specs:
            common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
            for name, spec in hidden_specs.items():
                per_token = (
                    spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                )
                new_bs = max(common_page // per_token, 1)
                aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
                groups.append(KVCacheGroupSpec([name], aligned))

        return groups

    def get_kv_cache_config_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """
        Generate the KV cache configuration from the KV cache groups and spec
        of each layer.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_groups: The KV cache groups
            available_memory: Memory available for KV cache in bytes
        Returns:
            The generated KVCacheConfig
        """
        if len(kv_cache_groups) == 0:
            # Attention free models do not have KV cache.
            # Return num_blocks=1 as BlockPool always needs a null_block.
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        # Determine how model runners should initialize the KV cache tensors.
        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            # Special case: all layers have the same type of KV cache but with
            # different hidden sizes. Allocate different amount of memory for each
            # layer based on its hidden size.
            num_blocks = (
                available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            num_blocks = may_override_num_blocks(self.vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
        else:
            # General case:
            # We will have group_size memory pools, each is shared by one layer from
            # each group. As layers of different groups have different block table,
            # they will use different parts of the shared Tensor.
            # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
            # (sw.1, padding) will be: (group_size = 2)
            # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
            # full.1, sw.2: share another Tensor with size=available_memory//2
            group_size = max(len(group.layer_names) for group in kv_cache_groups)

            page_size = get_uniform_page_size(
                [group.kv_cache_spec for group in kv_cache_groups]
            )
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = get_num_blocks(
                self.vllm_config, group_size, available_memory, page_size
            )
            kv_cache_tensors = []
            for i in range(group_size):
                shared_by = []
                for j in range(len(kv_cache_groups)):
                    if i < len(kv_cache_groups[j].layer_names):
                        shared_by.append(kv_cache_groups[j].layer_names[i])
                kv_cache_tensors.append(
                    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
                )

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )


    def _max_memory_usage_bytes_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        """
        Calculate maximum memory usage in bytes from KV cache groups.

        This correctly accounts for padding in hybrid models. For example, if a
        model has 8 full attention layers and 9 sliding window layers, they will
        be padded to 9 full + 9 sliding window for uniform group sizes.
        """
        if not kv_cache_groups:
            return 0

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            return sum(
                spec.max_memory_usage_bytes(self.vllm_config)
                for spec in per_layer_specs.values()
            )

        # General case: group_size pools, each shared by one layer per group
        # Memory = group_size * page_size * blocks_for_max_len
        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        blocks_needed = sum(
            cdiv(group.kv_cache_spec.max_memory_usage_bytes(self.vllm_config), page_size)
            for group in kv_cache_groups
        )

        return group_size * page_size * blocks_needed

    def get_max_model_len_capacity(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> int:
        # TODO(mengqing): rename me?
        return estimate_max_model_len_from_groups(
            self.vllm_config,
            kv_cache_groups,
            available_memory,
            self._max_memory_usage_bytes_from_groups,
        )

    def _auto_fit_max_model_len(
        self,
        projected_groups_per_worker: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> None:
        """
        When max_model_len is set to -1, this function estimates the largest
        context length that can be supported with the available GPU memory.
        It uses binary search to find the maximum length that fits across all
        workers.

        Args:
            vllm_config: The global VllmConfig (will be modified in-place)
            projected_groups_per_worker: KV cache groups projected to each worker.
            available_memory: Memory available for KV cache in bytes for each
                worker.
        """
        original_max = self.vllm_config.model_config.max_model_len

        if all(not groups for groups in projected_groups_per_worker):
            # All workers have empty specs (attention-free model)
            logger.info_once(
                "Auto-fit max_model_len: attention-free model, "
                "using derived max_model_len=%d",
                original_max,
            )
            return

        # Find the max_model_len that fits across all workers.
        auto_fit_max = original_max
        limiting_worker_mem = available_memory[0]
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                continue
            worker_max = self.get_max_model_len_capacity(groups, avail_mem)
            if worker_max < auto_fit_max:
                auto_fit_max = worker_max
                limiting_worker_mem = avail_mem

        if auto_fit_max <= 0:
            raise ValueError(
                "Cannot auto-fit max_model_len: not enough GPU memory available "
                "to serve even a single token. Try increasing `gpu_memory_utilization`."
            )

        if auto_fit_max >= original_max:
            # The model's full context length fits in memory
            logger.info_once(
                "Auto-fit max_model_len: full model context length %d fits in "
                "available GPU memory",
                original_max,
            )
        else:
            # Need to reduce max_model_len to fit in memory
            self.vllm_config.model_config.max_model_len = auto_fit_max
            logger.info_once(
                "Auto-fit max_model_len: reduced from %d to %d to fit in "
                "available GPU memory (%s GiB available for KV cache)",
                original_max,
                auto_fit_max,
                format_gib(limiting_worker_mem),
            )
