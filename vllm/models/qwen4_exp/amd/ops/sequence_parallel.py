# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


def _sp_shard_op(x: torch.Tensor) -> torch.Tensor:
    output = common_sp.sp_shard(x)
    # A functional custom op must not return a view of its input.
    unpadded = x.shape[0] % get_tensor_model_parallel_world_size() == 0
    return output.clone() if unpadded else output


def _sp_local_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return x.new_empty((cdiv(x.shape[0], tp_size), *x.shape[1:]))


def _sp_all_gather_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:])
    )


def _sp_router():
    """AITER's INT4 sequence-parallel collectives of the TP group, built by
    `decoder_layer_sp_collectives` before the first forward, or None."""
    qr_comm = getattr(get_tp_group().device_communicator, "qr_comm", None)
    return None if qr_comm is None else qr_comm.sp_collectives()


def _sp_reduce_scatter_int4(x: torch.Tensor) -> torch.Tensor:
    """AITER's INT4 reduce-scatter where it serves `x`, the common one elsewhere."""
    assert x.ndim == 2
    # A whole number of rows per rank, as the common reduce-scatter pads it.
    pad = -x.shape[0] % get_tensor_model_parallel_world_size()
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    router = _sp_router()
    output = None if router is None else router.reduce_scatter(x)
    return output if output is not None else common_sp.sp_reduce_scatter(x)


def _sp_all_gather_int4(x: torch.Tensor) -> torch.Tensor:
    """AITER's INT4 all-gather where it serves `x`, the common one elsewhere."""
    router = _sp_router()
    output = None if router is None else router.all_gather(x)
    return output if output is not None else common_sp.sp_all_gather(x)


# Dynamo would specialize the padding branch on the traced token count, and
# vLLM drops the resulting guards. Keep the branch opaque when compiling. AITER's
# launches are Python calls Dynamo cannot trace either.
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
direct_register_custom_op(
    op_name="sp_reduce_scatter_int4",
    op_func=_sp_reduce_scatter_int4,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_all_gather_int4",
    op_func=_sp_all_gather_int4,
    fake_impl=_sp_all_gather_fake,
)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter(x)
    return common_sp.sp_reduce_scatter(x)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_shard(x)
    return common_sp.sp_shard(x)


def sp_reduce_scatter_int4(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter_int4(x)
    return _sp_reduce_scatter_int4(x)


def sp_all_gather_int4(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_all_gather_int4(x)
    return _sp_all_gather_int4(x)


def decoder_layer_sp_collectives(
    vllm_config: VllmConfig, hc_sp: bool
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]:
    """The (all_gather, reduce_scatter) pair of the decoder layer's four SP sites.

    With `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` and
    `--enable-hc-sp`, AITER's INT4 collectives take the payloads its window for
    this platform covers, and the common pair runs the rest. AITER decides, so a
    platform or an AITER without them gets the common pair. The first call
    builds them on the TP group, which is a collective, so every layer
    constructor makes it. Only the explicit flag opts in: MoE sequence
    parallelism, which also turns HC SP on, keeps the exact pair. DBO and
    microbatching (`use_ubatching`) are excluded: the INT4 launches need one
    serialized device order. The other SP
    sites (embedding shard, LM-head gather, PLE) always stay on the common path.
    """
    parallel_config = vllm_config.parallel_config
    if (
        hc_sp
        and parallel_config.enable_hc_sp
        and not parallel_config.use_ubatching
        and _sp_router() is not None
    ):
        return sp_all_gather_int4, sp_reduce_scatter_int4
    return common_sp.sp_all_gather, sp_reduce_scatter
