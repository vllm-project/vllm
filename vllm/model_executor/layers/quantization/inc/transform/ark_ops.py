# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..schemes.inc_ark_ops import get_ark_state

logger = init_logger(__name__)

_OPS_REGISTERED = False


def _inc_ark_mxfp4_hadamard_quant_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from auto_round_kernel.mxfp4_hadamard import mxfp4_hadamard_quant

    return mxfp4_hadamard_quant(x)


def _inc_ark_mxfp4_hadamard_quant_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8, device=x.device
        ),
        torch.empty(
            (*x.shape[:-1], x.shape[-1] // 32), dtype=torch.uint8, device=x.device
        ),
    )


def register_ops_once() -> None:
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    is_available, error_str, _, _ = get_ark_state()
    if not is_available:
        logger.debug(
            "Skip registering inc_ark_mxfp4_hadamard_quant because ARK is "
            "unavailable: %s",
            error_str or "unknown error",
        )
        return

    direct_register_custom_op(
        op_name="inc_ark_mxfp4_hadamard_quant",
        op_func=_inc_ark_mxfp4_hadamard_quant_impl,
        fake_impl=_inc_ark_mxfp4_hadamard_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _OPS_REGISTERED = True


register_ops_once()

__all__ = ["register_ops_once"]
