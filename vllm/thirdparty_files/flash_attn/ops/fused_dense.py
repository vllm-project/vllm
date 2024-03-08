# Copyright (c) 2023, Tri Dao.
# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# We make it work with pytorch amp and with bfloat16.
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py
from functools import partial
from typing import Optional

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
    all_gather_raw,
    all_reduce,
    all_reduce_raw,
    reduce_scatter,
    reduce_scatter_raw,
)


class FusedDenseFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, x, weight, bias, return_residual=False, process_group=None, sequence_parallel=True
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_output, weight
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel:
                handle_x.wait()
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(
            x, weight, bias, return_residual, process_group, sequence_parallel
        )
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)


class FusedDense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_residual: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.
        """
        return fused_dense_func(
            x,
            self.weight,
            self.bias,
            return_residual=self.return_residual,
            process_group=process_group,
        )


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        super().__init__(
            in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return fused_dense_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
        )


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        # Only rank 0 will have bias
        super().__init__(
            local_multiple * multiple_of,
            out_features,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class FusedMLPFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight1,
        bias1,
        weight2,
        bias2,
        activation="gelu_approx",
        save_pre_act=True,
        return_residual=False,
        checkpoint_lvl=0,
        heuristic=0,
        process_group=None,
        sequence_parallel=True,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather of x before doing the matmul.
        If sequence_parallel=False, then the input is already gathered.

        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out / relu_out in the bwd
        2: recompute pre_act and gelu_out / relu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        if activation == "sqrelu":
            assert heuristic == -1
        if not save_pre_act:
            checkpoint_lvl = 2
        assert checkpoint_lvl in [0, 1, 2]
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.activation = activation
        ctx.heuristic = heuristic

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2 = [a.to(dtype=dtype) for a in [weight1, weight2]]
            bias1 = bias1.to(dtype=dtype) if bias1 is not None else None
            bias2 = bias2.to(dtype=dtype) if bias2 is not None else None
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous() if bias1 is not None else None
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous() if bias2 is not None else None
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight1.shape, *weight2.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        if heuristic == -1:
            pre_act = F.linear(total_x, weight1, bias1)
            activation_fn = (
                partial(F.gelu, approximate="tanh")
                if activation == "gelu_approx"
                else (sqrelu_fwd if activation == "sqrelu" else F.relu)
            )
            with torch.jit.fuser("fuser2"):
                output1 = activation_fn(pre_act)
            # This is before adding bias1
            # pre_act = F.linear(total_x.reshape(batch_dim, n), weight1)
            # with torch.jit.fuser('fuser2'):
            #     output1 = bias_gelu(pre_act, bias1)
        else:
            is_gelu = activation == "gelu_approx"
            output1, *rest = fused_dense_cuda.linear_act_forward(
                total_x.reshape(batch_dim, n), weight1, bias1, is_gelu, save_pre_act, heuristic
            )
            if save_pre_act:
                pre_act = rest[0]
        output2 = F.linear(output1, weight2, bias2)
        if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == "relu"):
            # For RELU the pre_act is very small (just a bit-mask) so we just save it
            ctx.save_for_backward(x, weight1, weight2, pre_act, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, weight2, pre_act)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        output2 = output2.reshape(*batch_shape, output2.shape[-1])
        return output2 if not return_residual else (output2, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        activation = ctx.activation
        activation_fn = (
            partial(F.gelu, approximate="tanh")
            if activation == "gelu_approx"
            else (sqrelu_fwd if activation == "sqrelu" else F.relu)
        )
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        x, weight1, weight2, *rest = ctx.saved_tensors
        if process_group is None or not sequence_parallel:
            total_x = x
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl in [0, 1]:
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == "relu"):
                pre_act, output1 = rest
            elif checkpoint_lvl == 1:
                (pre_act,) = rest
                with torch.jit.fuser("fuser2"):
                    output1 = activation_fn(pre_act)
        elif checkpoint_lvl == 2:
            (bias1,) = rest
            if process_group is not None and sequence_parallel:
                total_x, _ = all_gather_raw(x, process_group)
            if ctx.heuristic == -1:
                pre_act = F.linear(total_x, weight1, bias1)
                with torch.jit.fuser("fuser2"):
                    output1 = activation_fn(pre_act)
            else:
                output1, pre_act = fused_dense_cuda.linear_act_forward(
                    total_x.reshape(batch_dim, total_x.shape[-1]),
                    weight1,
                    bias1,
                    activation == "gelu_approx",
                    True,
                    ctx.heuristic,
                )

        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        output1 = output1.reshape(batch_dim, output1.shape[-1])
        pre_act = pre_act.reshape(batch_dim, pre_act.shape[-1])
        if ctx.needs_input_grad[3]:
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(
                output1, grad_output, ctx.needs_input_grad[4]
            )
        else:
            grad_weight2 = None
            grad_bias2 = grad_output if ctx.needs_input_grad[4] else None
        if ctx.heuristic == -1:
            # grad_pre_act = matmul_dgelu(grad_output, weight2, pre_act)
            grad_output1 = F.linear(grad_output, weight2.t())
            activation_grad_fn = (
                gelu_bwd
                if activation == "gelu_approx"
                else (sqrelu_bwd if activation == "sqrelu" else relu_bwd)
            )
            with torch.jit.fuser("fuser2"):
                grad_pre_act = activation_grad_fn(grad_output1, pre_act)
        else:
            # The cublasLt epilogue has to compute both gelu/relu grad and bias grad, we can't
            # just compute gelu/relu grad
            grad_pre_act, grad_bias1 = fused_dense_cuda.bias_act_linear_dgrad_bgrad(
                weight2, grad_output, pre_act, activation == "gelu_approx", ctx.heuristic
            )
            if not ctx.needs_input_grad[2]:
                grad_bias1 = None
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_pre_act, weight1.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_pre_act, weight1
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.heuristic == -1:
            if ctx.needs_input_grad[1]:
                if process_group is not None and sequence_parallel and checkpoint_lvl != 2:
                    handle_x.wait()
                grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_wgrad(
                    total_x.reshape(batch_dim, total_x.shape[-1]),
                    grad_pre_act,
                    ctx.needs_input_grad[2],
                )
            else:
                grad_weight1 = None
                grad_bias1 = grad_pre_act if ctx.needs_input_grad[2] else None
        else:
            if ctx.needs_input_grad[1]:
                if process_group is not None and sequence_parallel and checkpoint_lvl != 2:
                    handle_x.wait()
                grad_weight1 = F.linear(
                    grad_pre_act.t(), total_x.reshape(batch_dim, total_x.shape[-1]).t()
                )
            else:
                grad_weight1 = None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return (
            grad_input,
            grad_weight1,
            grad_bias1,
            grad_weight2,
            grad_bias2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_mlp_func(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    activation: str = "gelu_approx",
    save_pre_act: bool = True,
    return_residual: bool = False,
    checkpoint_lvl: int = 0,
    heuristic: int = 0,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    assert activation in ["gelu_approx", "relu", "sqrelu"]
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    # If we save pre-activation, dimension must be divisible by 128 (relu) or 8 (gelu)
    dim_eligible = not save_pre_act or (x.shape[-1] % (128 if activation == "relu" else 8) == 0)
    if (
        x.is_cuda
        and weight1.is_cuda
        and weight2.is_cuda
        and (bias1 is None or bias1.is_cuda)
        and (bias2 is None or bias2.is_cuda)
        and dtype_eligible
        and dim_eligible
    ):
        return FusedMLPFunc.apply(
            x,
            weight1,
            bias1,
            weight2,
            bias2,
            activation,
            save_pre_act,
            return_residual,
            checkpoint_lvl,
            heuristic,
            process_group,
            sequence_parallel,
        )
    else:
        assert process_group is None
        pre_act = F.linear(x, weight1, bias1)
        activation_fn = (
            partial(F.gelu, approximate="tanh")
            if activation == "gelu_approx"
            else partial(F.relu, inplace=True)
        )
        output1 = activation_fn(pre_act)
        output2 = F.linear(output1, weight2, bias2)
        return output2 if not return_residual else (output2, x)


class FusedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=True,
        bias2=True,
        activation="gelu_approx",
        return_residual=False,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    ):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
                For H100, we set heuristic=-1 for both fp16 and bf16 as the fused cuBlasLt implementation
                is slower than the unfused version.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.activation = activation
        self.return_residual = return_residual
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic if activation != "sqrelu" else -1
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x, process_group=None):
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        if self.heuristic == "auto":
            if self.activation == "gelu_approx":
                if torch.cuda.get_device_capability("cuda") == (9, 0):
                    heuristic = -1
                else:
                    cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                    heuristic = 0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        out = fused_mlp_func(
            x,
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            activation=self.activation,
            save_pre_act=self.training,
            return_residual=self.return_residual,
            checkpoint_lvl=self.checkpoint_lvl,
            heuristic=heuristic,
            process_group=process_group,
        )
        if self.return_residual:
            out, x = out
        if process_group is not None:
            out = reduce_scatter(out, process_group)
        return out if not self.return_residual else (out, x)


class ParallelFusedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation="gelu_approx",
        process_group: ProcessGroup = None,
        bias1=True,
        bias2=True,
        sequence_parallel=True,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    ):
        """
        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        assert process_group is not None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.activation = activation
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic if activation != "sqrelu" else -1
        self.fc1 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias=bias1, **factory_kwargs
        )
        self.fc2 = RowParallelLinear(
            hidden_features, out_features, process_group, bias=bias2, **factory_kwargs
        )

    def forward(self, x):
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        if self.heuristic == "auto":
            if self.activation == "gelu_approx":
                cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                heuristic = 0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        out = fused_mlp_func(
            x,
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            activation=self.activation,
            save_pre_act=self.training,
            checkpoint_lvl=self.checkpoint_lvl,
            heuristic=heuristic,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
        )
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)
