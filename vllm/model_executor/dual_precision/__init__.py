# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-precision residency: a BF16 model plus a low-precision shadow store
(GPTQ-packed INT4 or ModelOpt NVFP4), with every LoRA wrapper bound to one of
them per forward.

Public surface consumed by the model runner, the CUDA-graph dispatcher and
the scheduler-side switch:

* :func:`dual_precision_rollout_enabled`
* :func:`check_dual_precision_model_runner` (worker init: V1 runner only)
* :func:`attach_dual_precision` (once, after LoRA load)
* :func:`bind_dual_precision` (before every capture / replay / eager forward)
* :func:`mark_lifecycle_event` (worker sleep/wake-up, runner weight reload;
  arms a shadow re-validation at the next INT4 bind)
* :func:`get_active_precision`
* :data:`BASE_PRECISION_BF16`, :data:`BASE_PRECISION_INT4`,
  :data:`SHADOW_MODULE_NAME`
"""

from vllm.model_executor.dual_precision.binding import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    BASE_PRECISIONS,
    DualPrecisionBinding,
    DualPrecisionState,
    bind_dual_precision,
    get_active_base_layer,
    get_active_precision,
    get_binding,
    get_dual_precision_state,
    mark_lifecycle_event,
    set_analysis_bf16_layers,
)
from vllm.model_executor.dual_precision.loader import (
    SHADOW_MODULE_NAME,
    Int4ShadowLayerStore,
    attach_dual_precision,
    check_dual_precision_model_runner,
    dual_precision_rollout_enabled,
)

__all__ = [
    "BASE_PRECISION_BF16",
    "BASE_PRECISION_INT4",
    "BASE_PRECISIONS",
    "SHADOW_MODULE_NAME",
    "DualPrecisionBinding",
    "DualPrecisionState",
    "Int4ShadowLayerStore",
    "attach_dual_precision",
    "bind_dual_precision",
    "check_dual_precision_model_runner",
    "dual_precision_rollout_enabled",
    "get_active_base_layer",
    "get_active_precision",
    "get_binding",
    "get_dual_precision_state",
    "mark_lifecycle_event",
    "set_analysis_bf16_layers",
]
