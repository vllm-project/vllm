# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 KV cache layout planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from vllm.logger import init_logger
from vllm.model_executor.kv_cache_plan.model_kv_cache_planner import (
    DefaultModelKVCachePlanner,
)
from vllm.models.deepseek_v4.nvidia.kv_cache_planner_utils import (
    may_override_num_blocks,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.models.deepseek_v4.nvidia.kv_cache_planner_utils import estimate_max_model_len_from_groups
logger = init_logger(__name__)


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


class DeepseekV4KVCachePlanner(DefaultModelKVCachePlanner):
    """Plan DeepSeek V4 KV cache layouts and groups.

    DeepSeek V4 has several auxiliary caches whose natural page sizes differ
    from the main MLA cache. This planner assigns compatible block sizes and
    padded page sizes, then builds KV cache groups around canonical page-size
    buckets shared by the main MLA layers.
    """

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
        # TODO(mengqing): refresh the comment here to reflect the current logic.

        # all groups are UniformTypeKVCacheSpecs.
        # They must already be page_size aligned and share a common padded
        # layer-tuple layout. Even groups with fewer actual tuples still reserve
        # the global number of tuple slots in the shared tensor layout.
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )

        total_max_mem_usage_bytes = 0
        for group in kv_cache_groups:
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(self.vllm_config)
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes
        return total_max_mem_usage_bytes

    def _get_kv_cache_config(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> tuple[int, list[KVCacheTensor]]:
        """DeepseekV4 KV cache tensor layout planning.

        Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
        define the canonical bucket set. Non-full-MLA groups must have been
        page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
        every layer's page_size matches one of the full-MLA bucket sizes.

        For each group, bucket its layers by page_size_bytes and place each
        layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
        per (tuple_idx, bucket) whose shared_by is the union of per-group
        layers at that slot.
        """
        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
        page_sizes = sorted(full_mla_spec.get_page_sizes())
        layer_tuple_page_bytes = sum(page_sizes)

        # Pre-bucket each group's layers by page_size (registration order within
        # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
        bucketed: list[dict[int, list[str]]] = []
        for group in kv_cache_groups:
            assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            specs = group.kv_cache_spec.kv_cache_specs
            b: dict[int, list[str]] = defaultdict(list)
            for name in group.layer_names:
                b[specs[name].page_size_bytes].append(name)
            bucketed.append(b)

        # num_layer_tuples = longest bucket list across all groups. For the
        # full-MLA group this equals the count of layers in the largest
        # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
        # this equals the sub-group size (each has a single page_size).
        num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values())

        num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
        num_blocks = may_override_num_blocks(self.vllm_config, num_blocks)

        kv_cache_tensors: list[KVCacheTensor] = []
        for tuple_idx in range(num_layer_tuples):
            for ps in page_sizes:
                shared_by: list[str] = []
                for b in bucketed:
                    bucket = b.get(ps)
                    if bucket is not None and tuple_idx < len(bucket):
                        shared_by.append(bucket[tuple_idx])
                kv_cache_tensors.append(
                    KVCacheTensor(size=ps * num_blocks, shared_by=shared_by)
                )

        return num_blocks, kv_cache_tensors

    def get_kv_cache_config_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        if not kv_cache_groups:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )
        num_blocks, kv_cache_tensors = self._get_kv_cache_config(
            kv_cache_groups, available_memory
        )
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def get_kv_cache_groups(
        self,
        kv_cache_specs: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        # step 1: group the kv cache specs into mla group and swa groups
        grouped_specs = self._group_and_unify_kv_cache_specs(kv_cache_specs)
        if grouped_specs is None:
            raise ValueError(
                "DeepSeek V4 KV cache planner requires SlidingWindowMLASpec "
                "layers in the worker KV cache specs."
            )
        # step 2: generate the kv cache group according to the grouped_specs
        kv_cache_groups = self._get_kv_cache_groups_from_uniform_groups(grouped_specs)
        # step 3: annote eagle groups
        self._annotate_eagle_groups(kv_cache_specs, kv_cache_groups)
        return kv_cache_groups

    def _group_and_unify_kv_cache_specs(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[UniformTypeKVCacheSpecs] | None:
        """
        Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
        Currently, this is only used for DeepseekV4.
        """
        if not any(
            isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_specs.values()
        ):
            return None

        mla_specs: dict[str, KVCacheSpec] = {}
        grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = (
            defaultdict(dict)
        )
        # NOTE: Here we group SWA layers by (block_size, sliding_window), which separates
        # SWA layers, C4I+C4A layers, and C128A layers into three different groups. It can
        # be fragile with only block_size and sliding_window as keys, but fine for now.
        for name, spec in kv_cache_specs.items():
            if isinstance(spec, SlidingWindowMLASpec):
                grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = (
                    spec
                )
            elif isinstance(spec, MLAAttentionSpec):
                mla_specs[name] = spec

        assert len(mla_specs) > 0
        mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
        assert mla_uniform_spec is not None

        swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
        for spec_dict in grouped_swa_mla_specs.values():
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
            assert uniform_spec is not None
            swa_uniform_specs.append(uniform_spec)

        return [mla_uniform_spec, *swa_uniform_specs]

    def _get_kv_cache_groups_from_uniform_groups(
        self, grouped_specs: list[UniformTypeKVCacheSpecs]
    ) -> list[KVCacheGroupSpec]:
        """
        Generate the KV cache groups from the grouped specs.
        """
        assert len(grouped_specs) > 0 and all(
            isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
        )
        # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
        # containing only MLAAttentionSpec.
        full_mla_spec = grouped_specs[0]
        assert all(
            isinstance(spec, MLAAttentionSpec)
            for spec in full_mla_spec.kv_cache_specs.values()
        )
        full_mla_group = KVCacheGroupSpec(
            layer_names=list(full_mla_spec.kv_cache_specs.keys()),
            kv_cache_spec=full_mla_spec,
        )

        # We define a layer tuple as a group of layers with different page sizes, and
        # one UniformTypeKVCacheSpecs contains a list of layer tuples.
        # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
        # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
        # The other uniform KV cache specs will be similarly partitioned into layer tuples.
        # Say we have 21 SWA layers, all with the same page size, then we will have "21"
        # layer tuples.
        num_layer_tuples_per_group: list[int] = [
            g_spec.get_num_layer_tuples() for g_spec in grouped_specs
        ]
        # Choose `num_layer_tuples` to minimize total padding across groups.
        num_layer_tuples = _approximate_gcd(
            num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
        )
        # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
        num_layer_tuples_per_group = [
            round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
        ]

        swa_mla_specs = grouped_specs[1:]
        assert all(
            isinstance(spec, SlidingWindowMLASpec)
            for group in swa_mla_specs
            for spec in group.kv_cache_specs.values()
        )

        # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
        # Possibly padding layer tuples for this.
        # Additionally, we also pad KV blocks in each SWA layer, to align the page size
        # with the corresponding layer in the full-MLA group.
        all_page_sizes = full_mla_spec.get_page_sizes()
        swa_mla_groups = []
        for sm_spec in swa_mla_specs:
            sm_page_sizes = sm_spec.get_page_sizes()
            layers_per_size: dict[int, list[str]] = defaultdict(list)
            assert max(sm_page_sizes) <= max(all_page_sizes)

            # Unify page size by padding layers' page_size to the nearest larger page_size.
            # Compute candidate (nearest larger page_size) for each unique page size.
            size_to_candidate: dict[int, int] = {}
            for ps in sm_page_sizes:
                size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
            # Pad and collect layer names per page size.
            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                current_size = layer_spec.page_size_bytes
                candidate = size_to_candidate[current_size]
                if current_size < candidate:
                    object.__setattr__(layer_spec, "page_size_padded", candidate)
                layers_per_size[candidate].append(layer_name)
            # NOTE(yifan): for now, inside a UniformKV group, each page_size should
            # have the same number of layers. This also means we don't need to pad layers
            # inside a partial-full layer tuple.
            assert len(set(len(layers) for layers in layers_per_size.values())) == 1
            num_layers_per_size = len(next(iter(layers_per_size.values())))

            # Split layers inside each UniformKV group for aligned #(layers).
            # See `_get_kv_cache_groups_uniform_page_size` for more details.
            num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
            layer_tuples = list(zip(*layers_per_size.values()))
            for i in range(num_tuple_groups):
                group_layer_tuples = layer_tuples[i::num_tuple_groups]
                # Flatten tuples and build dict for from_specs
                group_layer_names = [
                    name for layer_tuple in group_layer_tuples for name in layer_tuple
                ]
                group_layer_specs = {
                    name: sm_spec.kv_cache_specs[name] for name in group_layer_names
                }
                sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
                assert sub_sm_spec is not None
                swa_mla_groups.append(
                    KVCacheGroupSpec(
                        layer_names=group_layer_names,
                        kv_cache_spec=sub_sm_spec,
                    )
                )

        return [full_mla_group, *swa_mla_groups]

    def _annotate_eagle_groups(
        self,
        kv_cache_spec: dict[str, KVCacheSpec],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        spec_config = self.vllm_config.speculative_config
        if spec_config is None or not spec_config.use_eagle():
            return
        # Detection uses the merged MLA spec's model_version.
        if not any(
            getattr(spec, "model_version", None) == "deepseek_v4"
            for spec in kv_cache_spec.values()
        ):
            return
        # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
        # group contains it.
        # FIXME(yifan): avoid/generalize this hacky check.
        last_layer = next(reversed(kv_cache_spec))
        for group in kv_cache_groups:
            if last_layer in group.layer_names:
                group.is_eagle_group = True
                break