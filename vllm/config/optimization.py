# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.model import ModelConfig


class OptimizationLevel(Enum):
    """Optimization level enum."""

    O0 = 0
    """00 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation but no
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as cudagraphs."""
    O3 = 3
    """O3: Full (auto)tuning. -O2 as well as max-autotune, compiling for
    additional static sizes, etc. - any other time-consuming optimizations."""


def build_defaults(
    optimization_level: OptimizationLevel,
    model_config: ModelConfig | None = None,
):
    is_quantized = False
    is_sequential = False
    if model_config is not None:
        is_quantized = model_config.is_quantized()
        is_sequential = not model_config.is_model_moe()
    optimization_level_00 = {
        "general": {
            "pass_config": {
                "enable_noop": False,
                "enable_fusion": False,
                "enable_fi_allreduce_fusion": False,
            },
            "mode": CompilationMode.NONE,
            "cudagraph_mode": CUDAGraphMode.NONE,
            "use_inductor_graph_partition": False,
        },
        "is_quantized": {"pass_config": {"enable_attn_fusion": False}},
        "is_sequential": {
            "pass_config": {
                "enable_sequence_parallelism": False,
                "enable_async_tp": False,
            }
        },
    }
    optimization_level_01 = {
        "general": {
            "pass_config": {
                "enable_noop": True,
                "enable_fusion": True,
                "enable_fi_allreduce_fusion": False,
            },
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.PIECEWISE,
            "use_inductor_graph_partition": False,
        },
        "is_quantized": {"pass_config": {"enable_attn_fusion": False}},
        "is_sequential": {
            "pass_config": {
                "enable_sequence_parallelism": False,
                "enable_async_tp": False,
            }
        },
    }
    optimization_level_02 = {
        "general": {
            "pass_config": {
                "enable_noop": True,
                "enable_fusion": True,
                "enable_fi_allreduce_fusion": True,
            },
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
            "use_inductor_graph_partition": True,
        },
        "is_quantized": {"pass_config": {"enable_attn_fusion": is_quantized}},
        "is_sequential": {
            "pass_config": {
                "enable_sequence_parallelism": is_sequential,
                "enable_async_tp": is_sequential,
            }
        },
    }
    optimization_level_03 = {
        "general": {
            "pass_config": {
                "enable_noop": True,
                "enable_fusion": True,
                "enable_fi_allreduce_fusion": True,
            },
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
            "use_inductor_graph_partition": True,
        },
        "is_quantized": {"pass_config": {"enable_attn_fusion": is_quantized}},
        "is_sequential": {
            "pass_config": {
                "enable_sequence_parallelism": is_sequential,
                "enable_async_tp": is_sequential,
            }
        },
    }
    optimization_level_to_config = {
        OptimizationLevel.O0: optimization_level_00,
        OptimizationLevel.O1: optimization_level_01,
        OptimizationLevel.O2: optimization_level_02,
        OptimizationLevel.O3: optimization_level_03,
    }
    return optimization_level_to_config[optimization_level]
