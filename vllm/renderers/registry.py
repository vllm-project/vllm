# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import assert_never

from vllm.logger import init_logger
from vllm.transformers_utils.repo_utils import list_filtered_repo_files
from vllm.utils.import_utils import resolve_obj_by_qualname

from .protocol import RendererLike

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_T = TypeVar("_T", bound=type[RendererLike])


class RendererRegistry:
    # Renderer name ->  (renderer module, renderer class)
    REGISTRY: dict[str, tuple[str, str]] = {
        "deepseekv32": ("vllm.renderers.deepseekv32", "DeepseekV32Renderer"),
        "hf": ("vllm.renderers.hf", "HfRenderer"),
        "mistral": ("vllm.renderers.mistral", "MistralRenderer"),
        "terratorch": ("vllm.renderers.terratorch", "TerratorchRenderer"),
    }

    @staticmethod
    def register(renderer_mode: str, module: str, class_name: str) -> None:
        if renderer_mode in RendererRegistry.REGISTRY:
            logger.warning(
                "%s.%s is already registered for renderer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                renderer_mode,
            )

        RendererRegistry.REGISTRY[renderer_mode] = (module, class_name)

        return None

    @staticmethod
    def init_renderer(
        renderer_mode: str,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> RendererLike:
        if renderer_mode not in RendererRegistry.REGISTRY:
            raise ValueError(f"No renderer registered for {renderer_mode=!r}.")

        module, class_name = RendererRegistry.REGISTRY[renderer_mode]
        logger.debug_once(f"Loading {class_name} for {renderer_mode=!r}")

        cls_: type[RendererLike] = resolve_obj_by_qualname(f"{module}.{class_name}")
        return cls_.from_config(config, tokenizer_kwargs)


def renderer_from_config(config: "ModelConfig"):
    tokenizer_name = config.tokenizer
    tokenizer_mode = config.tokenizer_mode
    tokenizer_revision = config.tokenizer_revision
    trust_remote_code = config.trust_remote_code
    tokenizer_kwargs = dict[str, Any]()

    runner_type = config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        tokenizer_kwargs["truncation_side"] = "left"
    elif runner_type == "pooling":
        tokenizer_kwargs["truncation_side"] = "right"
    else:
        assert_never(runner_type)

    tokenizer_mode = config.tokenizer_mode
    if tokenizer_mode == "slow":
        if tokenizer_kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")

        tokenizer_mode = "hf"
        tokenizer_kwargs["use_fast"] = False

    # Try to use official Mistral tokenizer if possible
    if tokenizer_mode == "auto" and importlib.util.find_spec("mistral_common"):
        allow_patterns = ["tekken.json", "tokenizer.model.v*"]
        files_list = list_filtered_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=allow_patterns,
            revision=tokenizer_revision,
        )
        if len(files_list) > 0:
            tokenizer_mode = "mistral"

    # Fallback to HF tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"

    tokenizer_kwargs = dict[str, Any](
        trust_remote_code=trust_remote_code,
        revision=tokenizer_revision,
        **tokenizer_kwargs,
    )

    return RendererRegistry.init_renderer(
        tokenizer_mode,
        config,
        tokenizer_kwargs,
    )
