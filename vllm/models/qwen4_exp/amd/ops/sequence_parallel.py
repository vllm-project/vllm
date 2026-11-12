# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
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


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter(x)
    return common_sp.sp_reduce_scatter(x)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_shard(x)
    return common_sp.sp_shard(x)
