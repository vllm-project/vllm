# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _sp_shard_op(x: torch.Tensor) -> torch.Tensor:
    output = common_sp.sp_shard(x)
    # A functional custom op must not return a view of its input.
    unpadded = x.shape[0] % get_tensor_model_parallel_world_size() == 0
    return output.clone() if unpadded else output


def _sp_local_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return x.new_empty((cdiv(x.shape[0], tp_size), *x.shape[1:]))


# Dynamo would specialize the padding branch on the traced token count, and
# vLLM drops the resulting guards. Keep the branch opaque when compiling.
direct_register_custom_op(
    op_name="sp_reduce_scatter",
    op_func=common_sp.sp_reduce_scatter,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_shard",
    op_func=_sp_shard_op,
    fake_impl=_sp_local_fake,
)


def _sp_all_gather_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:])
    )


# The QuickReduce launch is a Python call into AITER, which Dynamo cannot trace.
direct_register_custom_op(
    op_name="sp_reduce_scatter_quantized",
    op_func=common_sp.sp_reduce_scatter_quantized,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_all_gather_quantized",
    op_func=common_sp.sp_all_gather_quantized,
    fake_impl=_sp_all_gather_fake,
)


def _aiter_comm():
    """The TP group's AITER custom all-reduce, or None when it has none or it is
    disabled. The TP communicator's own reduce-scatter / all-gather stay on RCCL
    (AITER's path there is only enabled for DP groups), so the helpers ask AITER
    itself. It is asked on every call: `disable_custom_all_reduce` is not part of
    the compile hash, so a cached graph may run where the communicator is absent."""
    from vllm._aiter_ops import rocm_aiter_ops

    comm = rocm_aiter_ops.get_aiter_allreduce()
    return None if comm is None or comm.disabled else comm


def use_sp_aiter_ipc(vllm_config: VllmConfig) -> bool:
    """Whether the decoder-layer SP collectives go through AITER's exact IPC
    reduce-scatter / all-gather: `--enable-hc-sp-aiter-ipc` is set, the GPU is
    gfx950 and the TP group has the AITER custom all-reduce. Warns once when the
    flag is set but the group cannot serve it."""
    if not vllm_config.parallel_config.enable_hc_sp_aiter_ipc:
        return False
    from vllm.platforms.rocm import get_cdna_version

    # gfx950: the only arch run
    if get_cdna_version() == 4 and _aiter_comm() is not None:
        logger.info_once(
            "Using AITER IPC reduce-scatter and all-gather for the Qwen4Exp "
            "decoder-layer sequence-parallel collectives."
        )
        return True
    logger.warning_once(
        "--enable-hc-sp-aiter-ipc needs gfx950, VLLM_ROCM_USE_AITER=1 and custom "
        "all-reduce; the sequence-parallel collectives stay on RCCL."
    )
    return False


def _pad_rows(x: torch.Tensor) -> torch.Tensor:
    """`x` padded to a whole number of rows per TP rank, as the common
    reduce-scatter pads it."""
    pad = -x.shape[0] % get_tensor_model_parallel_world_size()
    return F.pad(x, (0, 0, 0, pad)) if pad else x


def _quick_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    """The INT4 QuickReduce collective `name` of the TP communicator on `x`, or
    None when it is not enabled or `x` does not qualify."""
    collective = getattr(get_tp_group().device_communicator, name, None)
    return None if collective is None else collective(x)


def _sp_reduce_scatter_ipc(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    x = _pad_rows(x)
    comm = _aiter_comm()
    if comm is not None and comm.should_custom_rs(x, 0):
        out = torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
        comm.custom_reduce_scatter(x, out, dim=0)
        return out
    return common_sp.sp_reduce_scatter(x)


def _sp_all_gather_ipc(x: torch.Tensor) -> torch.Tensor:
    comm = _aiter_comm()
    if comm is not None and comm.should_custom_ag(x):
        return comm.custom_all_gather(x, dim=0)
    return common_sp.sp_all_gather(x)


def _sp_reduce_scatter_quantized_ipc(x: torch.Tensor) -> torch.Tensor:
    """INT4 QuickReduce where the shape qualifies, the IPC pair below its size
    gate."""
    assert x.ndim == 2
    x = _pad_rows(x)
    output = _quick_collective("quick_reduce_scatter", x)
    return output if output is not None else _sp_reduce_scatter_ipc(x)


def _sp_all_gather_quantized_ipc(x: torch.Tensor) -> torch.Tensor:
    output = _quick_collective("quick_all_gather", x)
    return output if output is not None else _sp_all_gather_ipc(x)


# AITER's launches are Python calls Dynamo cannot trace.
direct_register_custom_op(
    op_name="sp_reduce_scatter_ipc",
    op_func=_sp_reduce_scatter_ipc,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_all_gather_ipc",
    op_func=_sp_all_gather_ipc,
    fake_impl=_sp_all_gather_fake,
)
direct_register_custom_op(
    op_name="sp_reduce_scatter_quantized_ipc",
    op_func=_sp_reduce_scatter_quantized_ipc,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_all_gather_quantized_ipc",
    op_func=_sp_all_gather_quantized_ipc,
    fake_impl=_sp_all_gather_fake,
)


def sp_reduce_scatter_ipc(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter_ipc(x)
    return _sp_reduce_scatter_ipc(x)


def sp_all_gather_ipc(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_all_gather_ipc(x)
    return _sp_all_gather_ipc(x)


def sp_reduce_scatter_quantized_ipc(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter_quantized_ipc(x)
    return _sp_reduce_scatter_quantized_ipc(x)


def sp_all_gather_quantized_ipc(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_all_gather_quantized_ipc(x)
    return _sp_all_gather_quantized_ipc(x)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter(x)
    return common_sp.sp_reduce_scatter(x)


def sp_reduce_scatter_quantized(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter_quantized(x)
    return common_sp.sp_reduce_scatter_quantized(x)


def sp_all_gather_quantized(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_all_gather_quantized(x)
    return common_sp.sp_all_gather_quantized(x)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_shard(x)
    return common_sp.sp_shard(x)


# The decoder layer's four SP collectives as (all_gather, reduce_scatter), keyed by
# (INT4 QuickReduce, AITER IPC).
DECODER_LAYER_SP_COLLECTIVES = {
    (False, False): (common_sp.sp_all_gather, sp_reduce_scatter),
    (False, True): (sp_all_gather_ipc, sp_reduce_scatter_ipc),
    (True, False): (sp_all_gather_quantized, sp_reduce_scatter_quantized),
    (True, True): (sp_all_gather_quantized_ipc, sp_reduce_scatter_quantized_ipc),
}
