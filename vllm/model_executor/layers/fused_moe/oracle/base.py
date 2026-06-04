# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for MoE kernel selection oracles.

Every quantisation variant (FP8, NvFP4, MXFP4, INT8, unquantised, …) uses
an "oracle" that selects the best backend, converts weights and assembles the
final :class:`FusedMoEKernel`.  The concrete logic differs across quant
types, but the kernel-assembly step is virtually identical.

``MoEKernelOracle`` captures the shared logic so individual oracles can
inherit it instead of copy-pasting the same function.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

logger = init_logger(__name__)


class MoEKernelOracle:
    """Base helper for MoE kernel selection oracles.

    Subclasses override ``select_backend``, ``convert_weights``, and/or
    ``make_quant_config`` as needed.  The kernel assembly step
    (``make_moe_kernel``) is shared across **all** oracles.
    """

    # -----------------------------------------------------------------
    # Shared: kernel assembly
    # -----------------------------------------------------------------
    @staticmethod
    def make_moe_kernel(
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        *,
        experts_extra_kwargs: dict | None = None,
    ) -> mk.FusedMoEKernel:
        """Assemble a :class:`FusedMoEKernel` from *prepare/finalize* and
        *experts* components.

        This method is intentionally kept as a ``@staticmethod`` so that
        existing call-sites can migrate incrementally – they can call
        ``MoEKernelOracle.make_moe_kernel(…)`` without instantiating an
        oracle object.

        Args:
            quant_config: Quantisation configuration (scales, dtypes, …).
            moe_config: MoE layer configuration (parallelism, topology, …).
            experts_cls: Concrete expert implementation class.
            routing_tables: Optional pre-computed routing tables for EP.
            experts_extra_kwargs: Additional keyword arguments forwarded
                to the *experts_cls* constructor (e.g. ``b_strides`` for
                W4A8 CUTLASS, ``layer`` for Humming).
        """
        extra = experts_extra_kwargs or {}

        # --- Prepare / Finalize ---
        is_monolithic = issubclass(
            experts_cls, mk.FusedMoEExpertsMonolithic
        )
        prepare_finalize = maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            routing_tables=routing_tables,
            allow_new_interface=True,
            use_monolithic=is_monolithic,
        )
        assert prepare_finalize is not None

        logger.info_once("Using %s", prepare_finalize.__class__.__name__)

        # --- Experts ---
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens is not None
            experts = experts_cls(
                moe_config=moe_config,
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                **extra,
            )
        else:
            experts = experts_cls(
                moe_config=moe_config,
                quant_config=quant_config,
                **extra,
            )

        return mk.FusedMoEKernel(prepare_finalize, experts)
