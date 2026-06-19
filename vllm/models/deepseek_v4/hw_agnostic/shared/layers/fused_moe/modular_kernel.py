# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modular MoE kernel — re-exported from upstream.

The vendored DSv4 fused-MoE pipeline calls into upstream concrete
expert classes (``TritonExperts``, etc.) which inherit from upstream's
``FusedMoEExperts*`` / ``FusedMoEPrepareAndFinalize*`` ABCs. Several
vendored files do ``isinstance(...)`` / ``issubclass(...)`` checks
against those bases (see ``oracle/unquantized.py``,
``runner/moe_runner.py``, etc.). If the vendored module were a
separate copy, the upstream-returned objects would fail those checks.
Re-exporting upstream keeps both sides referencing the same ABCs.

A dedicated lint carve-out
(``model_executor.layers.fused_moe.modular_kernel``) lets this
through. Same justification as
``MoEActivation`` / ``QuantizationConfig`` / ``FusedMoEMethodBase``:
the public surface is pure ABCs and small data classes that have to
have identity-matching cross-boundary.
"""

from vllm.model_executor.layers.fused_moe.modular_kernel import *  # noqa: F401,F403
from vllm.model_executor.layers.fused_moe.modular_kernel import (  # noqa: F401
    ExpertTokensMetadata,
    FusedMoEActivationFormat,
    FusedMoEExperts,
    FusedMoEExpertsModular,
    FusedMoEExpertsMonolithic,
    FusedMoEKernel,
    FusedMoEKernelModularImpl,
    FusedMoEKernelMonolithicImpl,
    FusedMoEPrepareAndFinalize,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoEPrepareAndFinalizeMonolithic,
    TopKWeightAndReduce,
)
