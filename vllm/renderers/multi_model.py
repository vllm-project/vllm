# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model dispatch wrapper for the CPU-only render server.

The render server was originally scoped to one model per process. This wrapper
lets a single render process host N models — indexed by served-model-name —
while presenting the same surface as a single `OnlineRenderer` /
`OnlineDerenderer` to consumers via `.resolve(model_name)`.

Design intent (draft, RFC-scoped):
* Text and multimodal preprocessing scale linearly with N (tokenizer +
  processor per model). Each `OnlineRenderer` is independent — no cross-model
  state, no shared cache.
* Warmup fans out sequentially — parallel warmup is a follow-up if startup
  time becomes an issue.
* Consumers that pre-computed per-model state at construction time (e.g.
  `ServingRender.default_sampling_params`) must be refactored to resolve per
  request; this draft only wires up `/tokenize` and `/detokenize` end-to-end.
"""

from vllm.logger import init_logger
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer

logger = init_logger(__name__)


class MultiModelOnlineRenderer:
    """Dispatches to a per-model `OnlineRenderer` based on the request model.

    The `primary` renderer is the one built from `--model`; extra renderers
    come from `--extra-served-model`. Both are stored in `.renderers`, keyed
    by served name. `resolve(None)` returns `primary` so call sites that
    predate multi-model support keep working unchanged.
    """

    def __init__(
        self,
        primary_name: str,
        renderers: dict[str, OnlineRenderer],
    ) -> None:
        assert primary_name in renderers, (
            f"primary_name {primary_name!r} must be present in renderers "
            f"(got {sorted(renderers)})"
        )
        self.primary_name = primary_name
        self.renderers = renderers

    @property
    def primary(self) -> OnlineRenderer:
        return self.renderers[self.primary_name]

    def resolve(self, model_name: str | None) -> OnlineRenderer:
        if model_name and model_name in self.renderers:
            return self.renderers[model_name]
        return self.primary

    def warmup(self) -> None:
        for name, renderer in self.renderers.items():
            logger.info("Warming up renderer for model %r", name)
            renderer.warmup()


class MultiModelOnlineDerenderer:
    """Dispatches to a per-model `OnlineDerenderer` based on the request model."""

    def __init__(
        self,
        primary_name: str,
        derenderers: dict[str, OnlineDerenderer],
    ) -> None:
        assert primary_name in derenderers, (
            f"primary_name {primary_name!r} must be present in derenderers "
            f"(got {sorted(derenderers)})"
        )
        self.primary_name = primary_name
        self.derenderers = derenderers

    @property
    def primary(self) -> OnlineDerenderer:
        return self.derenderers[self.primary_name]

    def resolve(self, model_name: str | None) -> OnlineDerenderer:
        if model_name and model_name in self.derenderers:
            return self.derenderers[model_name]
        return self.primary
