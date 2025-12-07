# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.utils.import_utils import resolve_obj_by_qualname

from .protocol import RendererLike

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class RendererRegistry:
    def __init__(self) -> None:
        super().__init__()

        # Renderer name ->  (renderer module, renderer class)
        self._registry: dict[str, tuple[str, str]] = {}

    def register(self, renderer_mode: str, module: str, class_name: str) -> None:
        if renderer_mode in self._registry:
            logger.warning(
                "%s.%s is already registered for renderer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                renderer_mode,
            )

        self._registry[renderer_mode] = (module, class_name)

        return None

    def load_renderer_cls(self, renderer_mode: str) -> type[RendererLike]:
        if renderer_mode not in self._registry:
            raise ValueError(f"No renderer registered for {renderer_mode=!r}.")

        module, class_name = self._registry[renderer_mode]
        logger.debug_once(f"Loading {class_name} for {renderer_mode=!r}")

        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_renderer(
        self,
        renderer_mode: str,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> RendererLike:
        renderer_cls = self.load_renderer_cls(renderer_mode)
        return renderer_cls.from_config(config, tokenizer_kwargs)


RENDERER_REGISTRY = RendererRegistry()
"""The global `RendererRegistry` instance."""

RENDERER_REGISTRY._registry.update(
    {
        "deepseekv32": ("vllm.renderers.deepseekv32", "DeepseekV32Renderer"),
        "hf": ("vllm.renderers.hf", "HfRenderer"),
        "mistral": ("vllm.renderers.mistral", "MistralRenderer"),
        "terratorch": ("vllm.renderers.terratorch", "TerratorchRenderer"),
    }
)


def renderer_from_config(config: "ModelConfig", **kwargs):
    tokenizer_mode, tokenizer_args, tokenizer_kwargs = tokenizer_args_from_config(
        config, **kwargs
    )
    tokenizer_kwargs["tokenizer_name"] = tokenizer_args[0]

    if config.tokenizer_mode == "auto" and config.model_impl == "terratorch":
        renderer_mode = "terratorch"
    else:
        renderer_mode = tokenizer_mode

    return RENDERER_REGISTRY.load_renderer(renderer_mode, config, tokenizer_kwargs)
