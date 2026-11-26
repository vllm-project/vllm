# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GQA drafts (e.g. a Qwen3 sliding-window DSpark) on Kimi-K3: their KV cache grouping.

Dense MLA drafts keep their own MLA cache (`dense_draft.py`)."""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsKVCacheGrouping
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroup,
    KVCacheLayerConfig,
    KVCacheSpec,
    MLAAttentionSpec,
)

# A multiple of K3.1's 64-token prefix-hash unit. Draft blocks also divide the
# target's block, so the scheduler block (their LCM) does not change.
_DRAFT_BLOCK_ALIGN = 64


def _fit_draft_page(spec: KVCacheSpec, page_size: int) -> KVCacheSpec:
    """Shrink a draft's block until its page fits under `page_size`."""
    for block_size in range(spec.block_size, 0, -_DRAFT_BLOCK_ALIGN):
        fitted = replace(spec, block_size=block_size)
        if spec.block_size % block_size == 0 and fitted.page_size_bytes <= page_size:
            return fitted
    raise ValueError(
        f"No draft block size fits the target's {page_size}-byte KV page: {spec}"
    )


class GQADraftKVCacheGrouping(SupportsKVCacheGrouping):
    """Kimi-K3's KV cache grouping when a GQA draft is attached; otherwise the
    default grouping.

    Kimi-K3's own layers are KDA (Mamba) and MLA, so any other attention layer
    belongs to a draft. All groups share one page size and MLA pages cannot be
    padded, so the draft's page must fit under the target's. With all KV heads
    on one rank (attention TP1 under DP) it does not, and the draft gets the
    largest block that fits. Grouping the draft on its own also keeps the
    target's groups: the default grouping sizes every group by the draft's few
    layers (20 groups instead of 4 + 1 for a 5-layer draft).
    """

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: VllmConfig,
        layer_configs: dict[str, KVCacheLayerConfig],
    ) -> list[KVCacheGroup]:
        from vllm.v1.core.kv_cache_planning import (
            get_kv_cache_groups,
            unify_kv_cache_spec_page_size,
        )

        draft = {
            name: config
            for name, config in layer_configs.items()
            if isinstance(config.spec, AttentionSpec)
            and not isinstance(config.spec, MLAAttentionSpec)
        }
        target = {n: c for n, c in layer_configs.items() if n not in draft}
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not draft or not target or layout.is_block_outermost:
            return get_kv_cache_groups(vllm_config, layer_configs)
        page_size = max(c.spec.page_size_bytes for c in target.values())
        draft = {
            n: replace(c, spec=_fit_draft_page(c.spec, page_size))
            for n, c in draft.items()
        }
        configs = unify_kv_cache_spec_page_size(target | draft)
        return get_kv_cache_groups(
            vllm_config, {n: configs[n] for n in target}
        ) + get_kv_cache_groups(vllm_config, {n: configs[n] for n in draft})
