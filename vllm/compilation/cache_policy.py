# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


@dataclass(frozen=True)
class CompileCachePolicy:
    use_model_aot: bool
    use_mega_artifact: bool
    use_vllm_artifact_cache: bool

    @classmethod
    def resolve(
        cls,
        vllm_config: "VllmConfig | None" = None,
        *,
        model_config: "ModelConfig | None" = None,
    ) -> "CompileCachePolicy":
        from vllm.config.compilation import CompilationMode

        if model_config is None:
            current = _current_cache_policy.get()
            if current is not None and current[0] is vllm_config:
                return current[1]
            if vllm_config is not None:
                model_config = vllm_config.model_config
        if (
            vllm_config is not None
            and vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and model_config is not None
            and model_config.using_transformers_backend()
        ):
            # Re-trace HF's decorated Python code and let PyTorch validate the
            # resulting graphs. Graph-index handles bypass that validation.
            return cls(False, False, False)
        return cls(
            envs.VLLM_USE_AOT_COMPILE,
            envs.VLLM_USE_MEGA_AOT_ARTIFACT,
            True,
        )


_current_cache_policy: ContextVar[tuple["VllmConfig", CompileCachePolicy] | None] = (
    ContextVar("compile_cache_policy", default=None)
)


@contextmanager
def use_compile_cache_policy(
    vllm_config: "VllmConfig", policy: CompileCachePolicy
) -> Iterator[None]:
    """Bind the loader's model owner without replacing the target config."""
    token = _current_cache_policy.set((vllm_config, policy))
    try:
        yield
    finally:
        _current_cache_policy.reset(token)
