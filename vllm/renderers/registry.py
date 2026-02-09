# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.utils.import_utils import resolve_obj_by_qualname

from .protocol import BaseRenderer

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


_VLLM_RENDERERS = {
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Renderer"),
    "hf": ("hf", "HfRenderer"),
    "grok2": ("grok2", "Grok2Renderer"),
    "mistral": ("mistral", "MistralRenderer"),
    "terratorch": ("terratorch", "TerratorchRenderer"),
    "tiktoken": ("tiktoken", "TikTokenRenderer"),
}


@dataclass
class RendererRegistry:
    # Renderer mode ->  (renderer module, renderer class)
    renderers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, renderer_mode: str, module: str, class_name: str) -> None:
        if renderer_mode in self.renderers:
            logger.warning(
                "%s.%s is already registered for renderer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                renderer_mode,
            )

        self.renderers[renderer_mode] = (module, class_name)

        return None

    def load_renderer_cls(self, renderer_mode: str) -> type[BaseRenderer]:
        if renderer_mode not in self.renderers:
            raise ValueError(f"No renderer registered for {renderer_mode=!r}.")

        module, class_name = self.renderers[renderer_mode]
        logger.debug_once(f"Loading {class_name} for {renderer_mode=!r}")

        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_renderer(
        self,
        renderer_mode: str,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> BaseRenderer:
        renderer_cls = self.load_renderer_cls(renderer_mode)
        return renderer_cls.from_config(config, tokenizer_kwargs)


RENDERER_REGISTRY = RendererRegistry(
    {
        mode: (f"vllm.renderers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _VLLM_RENDERERS.items()
    }
)
"""The global `RendererRegistry` instance."""


def renderer_from_config(config: "ModelConfig", **kwargs):
    tokenizer_mode, tokenizer_name, args, kwargs = tokenizer_args_from_config(
        config, **kwargs
    )

    if config.tokenizer_mode == "auto" and config.model_impl == "terratorch":
        renderer_mode = "terratorch"
    else:
        renderer_mode = tokenizer_mode

    return RENDERER_REGISTRY.load_renderer(
        renderer_mode,
        config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )
