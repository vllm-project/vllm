# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_POST_GRAD_PASS_KEY = "post_grad_custom_post_pass"


class ZenCpuPlatform(CpuPlatform):
    """CPU platform with AMD Zen (ZenDNN/zentorch) optimizations.

    Model-load time (dispatch_cpu_unquantized_gemm in layers/utils.py):
      - Routes linear ops to zentorch_linear_unary.
      - When VLLM_ZENTORCH_WEIGHT_PREPACK=1 (default), eagerly prepacks
        weights via zentorch_weight_prepack_for_linear.

    Compile time:
      - Installs the Zen CPU fusion passes as Inductor's
        post_grad_custom_post_pass (see check_and_update_config).
    """

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        from vllm.config import CompilationMode

        compilation_config = vllm_config.compilation_config
        # CpuPlatform compiles in DYNAMO_TRACE_ONCE mode with the plain Inductor
        # backend, where PostGradPassManager does not run, so the passes go
        # straight into the Inductor config.
        if not (
            compilation_config.pass_config.fuse_zentorch_gated_act
            and compilation_config.mode == CompilationMode.DYNAMO_TRACE_ONCE
            and compilation_config.backend == "inductor"
        ):
            return

        from vllm.compilation.passes.fusion.zentorch_gated_act_fusion import (
            ZenCpuPostGradPass,
        )

        existing = compilation_config.inductor_compile_config.get(_POST_GRAD_PASS_KEY)
        if existing is None:
            passes = []
        elif isinstance(existing, (list, tuple)):
            passes = list(existing)
        else:
            passes = [existing]
        # check_and_update_config can run more than once per config.
        if any(isinstance(p, ZenCpuPostGradPass) for p in passes):
            return
        passes.append(ZenCpuPostGradPass(vllm_config))
        compilation_config.inductor_compile_config[_POST_GRAD_PASS_KEY] = (
            passes[0] if len(passes) == 1 else passes
        )

    device_name: str = "cpu"
    device_type: str = "cpu"

    def is_zen_cpu(self) -> bool:
        # is_cpu() also returns True for this platform (inherited from CpuPlatform).
        return True

    # fp16 compute needs AVX512-FP16, which AMD ships from Venice (6th Gen
    # EPYC) onwards; Turin (5th Gen EPYC) and earlier have no fp16 kernels.
    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if torch.ops.zentorch.zentorch_is_fp16_supported():
            return [torch.bfloat16, torch.float32, torch.float16]
        return [torch.bfloat16, torch.float32]
