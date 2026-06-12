# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import functools
import logging
import typing
from typing import Optional

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401


def _install_triton_fp8e4nv_pre_sm89_patch() -> None:
    """Install the fp8e4nv pre-SM89 Triton compatibility patch in memory.

    This mirrors the runtime behavior from Triton PR #10292 without rewriting
    Triton's installed files. vLLM can then compile kernels that pass fp8e4nv
    pointers opaquely on A100/SM80 and RTX 30xx/SM86, while actual fp8e4nv
    compute on those GPUs still fails with a clear compile-time error.
    """
    from collections.abc import Iterable
    from typing import Any

    logger = logging.getLogger(__name__)

    try:
        from triton.backends.nvidia import compiler as nvidia_compiler
    except Exception:
        logger.warning(
            "Could not import Triton NVIDIA compiler; fp8e4nv pointer use "
            "on SM80/SM86 may fail at Triton compile time.",
            exc_info=True,
        )
        return

    cuda_backend = getattr(nvidia_compiler, "CUDABackend", None)
    if cuda_backend is None:
        logger.warning(
            "Triton NVIDIA compiler has no CUDABackend; fp8e4nv pointer use "
            "on SM80/SM86 may fail at Triton compile time."
        )
        return

    patched_attr = "_vllm_fp8e4nv_pre_sm89_patch"

    def _add_item(items: Any, item: str) -> Any:
        if item in items:
            return items
        if isinstance(items, frozenset):
            return items | {item}
        if isinstance(items, set):
            updated = set(items)
            updated.add(item)
            return updated
        if isinstance(items, tuple):
            return (*items, item)
        if isinstance(items, list):
            return [*items, item]
        if isinstance(items, Iterable) and not isinstance(items, str):
            return (*tuple(items), item)
        return (item,)

    def _set_option(options: Any, name: str, value: Any) -> None:
        try:
            setattr(options, name, value)
        except (AttributeError, TypeError):
            object.__setattr__(options, name, value)

    def _convert_fp8e4nv_pre_sm89(
        arg: Any,
        dst_ty: Any,
        fp_downcast_rounding: Any = None,
        _semantic: Any = None,
    ) -> Any:
        supported = _semantic.builder.options.supported_fp8_dtypes
        raise ValueError(
            "type fp8e4nv not supported in this architecture for compute "
            "(no native or software conversion below compute capability 89). "
            f"The supported fp8 dtypes are {supported}. "
            "Pre-sm89 kernels may pass fp8e4nv pointers opaquely "
            "(load/store/pass-through) but cannot convert to/from fp8e4nv."
        )

    current_parse_options = getattr(cuda_backend, "parse_options", None)
    if current_parse_options is None:
        logger.warning(
            "CUDABackend has no parse_options; fp8e4nv pointer use on "
            "SM80/SM86 may fail at Triton compile time."
        )
        return

    if not getattr(current_parse_options, patched_attr, False):
        original_parse_options = current_parse_options

        @functools.wraps(original_parse_options)
        def parse_options_with_fp8e4nv_pre_sm89(self, opts) -> Any:
            options = original_parse_options(self, opts)
            fp8_dtypes = getattr(options, "supported_fp8_dtypes", ())
            _set_option(
                options,
                "supported_fp8_dtypes",
                _add_item(fp8_dtypes, "fp8e4nv"),
            )
            capability = int(self._parse_arch(options.arch))
            if capability < 89:
                deprecated = getattr(options, "deprecated_fp8_dot_operand_dtypes", ())
                _set_option(
                    options,
                    "deprecated_fp8_dot_operand_dtypes",
                    _add_item(deprecated, "fp8e4nv"),
                )
            return options

        setattr(parse_options_with_fp8e4nv_pre_sm89, patched_attr, True)
        cuda_backend.parse_options = parse_options_with_fp8e4nv_pre_sm89

    current_get_codegen = getattr(cuda_backend, "get_codegen_implementation", None)
    if current_get_codegen is None:
        logger.warning(
            "CUDABackend has no get_codegen_implementation; actual fp8e4nv "
            "compute on SM80/SM86 may fail with a lower-level Triton error."
        )
        return

    if not getattr(current_get_codegen, patched_attr, False):
        original_get_codegen = current_get_codegen

        @functools.wraps(original_get_codegen)
        def get_codegen_with_fp8e4nv_pre_sm89(self, options):
            codegen_fns = original_get_codegen(self, options)
            capability = int(self._parse_arch(options.arch))
            if capability < 89:
                codegen_fns["convert_fp8e4nv_pre_sm89"] = _convert_fp8e4nv_pre_sm89
            return codegen_fns

        setattr(get_codegen_with_fp8e4nv_pre_sm89, patched_attr, True)
        cuda_backend.get_codegen_implementation = get_codegen_with_fp8e4nv_pre_sm89

    try:
        import triton.language as tl
        from triton.language import semantic as triton_semantic
        from triton.language.semantic import TensorTy
    except Exception:
        logger.warning(
            "Could not import Triton semantic module; actual fp8e4nv compute "
            "on SM80/SM86 may fail with a lower-level Triton error.",
            exc_info=True,
        )
        return

    triton_semantic_cls = getattr(triton_semantic, "TritonSemantic", None)
    if triton_semantic_cls is None:
        logger.warning(
            "Triton semantic module has no TritonSemantic class; actual "
            "fp8e4nv compute on SM80/SM86 may fail with a lower-level "
            "Triton error."
        )
        return

    current_cast = getattr(triton_semantic_cls, "cast", None)
    if current_cast is None:
        logger.warning(
            "TritonSemantic has no cast method; actual fp8e4nv compute on "
            "SM80/SM86 may fail with a lower-level Triton error."
        )
        return

    if not getattr(current_cast, patched_attr, False):
        original_cast = current_cast

        @functools.wraps(original_cast)
        def cast_with_fp8e4nv_pre_sm89(
            self,
            input: TensorTy,
            dst_ty: tl.dtype,
            fp_downcast_rounding: Optional[str] = None,  # noqa: UP045
        ) -> TensorTy:
            src_ty = input.type
            normalized_dst_ty = dst_ty
            if src_ty.is_block():
                normalized_dst_ty = src_ty.with_element_ty(dst_ty.scalar)
            if src_ty != normalized_dst_ty:
                src_sca_ty = src_ty.scalar
                dst_sca_ty = normalized_dst_ty.scalar
                if src_sca_ty.is_fp8e4nv() or dst_sca_ty.is_fp8e4nv():
                    convert = self.builder.codegen_fns.get("convert_fp8e4nv_pre_sm89")
                    if convert is not None:
                        return convert(
                            input,
                            normalized_dst_ty,
                            fp_downcast_rounding,
                            _semantic=self,
                        )
            return original_cast(self, input, dst_ty, fp_downcast_rounding)

        setattr(cast_with_fp8e4nv_pre_sm89, patched_attr, True)
        triton_semantic_cls.cast = cast_with_fp8e4nv_pre_sm89

    logger.info("Installed in-memory Triton fp8e4nv pre-SM89 patch")


_install_triton_fp8e4nv_pre_sm89_patch()

MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
