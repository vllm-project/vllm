# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A speculative draft must keep a KV cache layout it can share with the target.

``get_kv_cache_spec()`` intersects layouts across every built backend, so a draft
that autoselects a backend with a disjoint layout set makes that intersection
empty and startup fails.
"""

import pytest
import torch

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_layout import KVCacheLayout


class _Backend:
    def __init__(self, layouts):
        self._layouts = layouts

    def supported_kv_cache_layouts(self):
        return self._layouts


def _backend(*names):
    return _Backend([KVCacheLayout[name] for name in names])


class _TargetLayer(AttentionLayerBase):
    """Minimal stand-in for a built target layer with a fixed backend."""

    def __init__(self, backend):
        self._backend = backend

    def get_attn_backend(self):
        return self._backend

    def get_kv_cache_spec(self, vllm_config):  # pragma: no cover - unused
        raise NotImplementedError


def _select_backend_for_draft(target_layouts: tuple[str, ...]) -> dict:
    """Record what the platform hook is asked for while building a draft."""
    captured: dict = {}

    def _spy(selected_backend, attn_selector_config, num_heads=None, **kwargs):
        captured["layout_constraint"] = kwargs.get("layout_constraint")
        return "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    config = VllmConfig()
    config.compilation_config.static_forward_context["target.attn"] = _TargetLayer(
        _backend(*target_layouts)
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention import selector

    selector._cached_get_attn_backend.cache_clear()
    original = current_platform.get_attn_backend_cls
    try:
        current_platform.get_attn_backend_cls = _spy
        with set_current_vllm_config(config), set_model_tag("eagle_head"):
            get_attn_backend(head_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    finally:
        current_platform.get_attn_backend_cls = original
        selector._cached_get_attn_backend.cache_clear()
    return captured


def test_draft_selection_is_constrained_to_the_targets_layouts() -> None:
    """The regression: without this the draft autoselects a disjoint layout."""
    captured = _select_backend_for_draft(("LBNHC",))
    assert captured["layout_constraint"] == ("LBNHC",)


def test_constraint_is_the_intersection_of_built_layers() -> None:
    captured = _select_backend_for_draft(("LBHNC", "BLHNC"))
    assert captured["layout_constraint"] == ("BLHNC", "LBHNC")


def test_backend_without_declared_layouts_always_fits() -> None:
    from vllm.v1.attention.backends.utils import backend_supports_layouts

    assert backend_supports_layouts(_Backend(None), ("LBNHC",))


def test_backend_fits_when_layouts_overlap() -> None:
    from vllm.v1.attention.backends.utils import backend_supports_layouts

    assert backend_supports_layouts(_backend("LBHNC", "BLHNC"), ("BLHNC",))


def test_backend_does_not_fit_when_layouts_are_disjoint() -> None:
    from vllm.v1.attention.backends.utils import backend_supports_layouts

    # The #54826 shape: a FLEX_ATTENTION target and a ROCM_ATTN style draft.
    assert not backend_supports_layouts(_backend("LHBNC", "LBHNC"), ("LBNHC",))


@pytest.mark.parametrize("constraint", [None, ()])
def test_absent_constraint_keeps_every_candidate(constraint) -> None:
    from vllm.v1.attention.backends.utils import drop_layout_incompatible_backends

    candidates = [_backend("LBNHC"), _backend("LHBNC")]
    assert (
        drop_layout_incompatible_backends(candidates, constraint, lambda c: c)
        is candidates
    )


def test_incompatible_candidates_are_dropped() -> None:
    from vllm.v1.attention.backends.utils import drop_layout_incompatible_backends

    keep = _backend("LBNHC")
    drop = _backend("LHBNC")
    kept = drop_layout_incompatible_backends([keep, drop], ("LBNHC",), lambda c: c)
    assert kept == [keep]


def test_constraint_is_ignored_when_it_would_leave_nothing() -> None:
    """Falls back so the caller reports its own error, not an empty list."""
    from vllm.v1.attention.backends.utils import drop_layout_incompatible_backends

    candidates = [_backend("LHBNC"), _backend("BLNHC")]
    assert (
        drop_layout_incompatible_backends(candidates, ("LBNHC",), lambda c: c)
        is candidates
    )
