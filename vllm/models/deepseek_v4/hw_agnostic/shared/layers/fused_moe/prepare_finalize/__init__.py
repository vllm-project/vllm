# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/finalize variants for the DSv4 hw-agnostic FusedMoE.

Vendored from
``vllm/model_executor/layers/fused_moe/prepare_finalize/`` with all
backends DSv4 doesn't exercise dropped:

  * ``batched`` / ``deepep_ht`` / ``deepep_ll`` / ``deepep_v2``
    (DeepEP variants) are not vendored.
  * ``flashinfer_nvlink_one_sided`` /
    ``flashinfer_nvlink_two_sided`` / ``mori`` / ``nixl_ep`` (NV / ROCm
    optional deps) are not vendored.
  * ``naive_dp_ep`` and ``no_dp_ep`` are kept — DSv4 with TP4_EP4 (or
    no DP) takes one of these two paths.
"""

from .naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
    MoEPrepareAndFinalizeNaiveDPEPMonolithic,
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from .no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
    MoEPrepareAndFinalizeNoDPEPMonolithic,
    make_moe_prepare_and_finalize_no_dp_ep,
)

__all__ = [
    "MoEPrepareAndFinalizeNaiveDPEPModular",
    "MoEPrepareAndFinalizeNaiveDPEPMonolithic",
    "make_moe_prepare_and_finalize_naive_dp_ep",
    "MoEPrepareAndFinalizeNoDPEPModular",
    "MoEPrepareAndFinalizeNoDPEPMonolithic",
    "make_moe_prepare_and_finalize_no_dp_ep",
]
