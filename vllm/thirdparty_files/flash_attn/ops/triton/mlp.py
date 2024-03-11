# The triton fused matmul + sqrelu is faster for fp16 but slower for bf16, compared
# to naive implementation.
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from flash_attn.ops.activations import sqrelu_bwd, sqrelu_fwd
from flash_attn.ops.triton.linear import triton_dgrad_act, triton_linear_act


class FusedDenseSqreluDenseFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_lvl=0):
        """checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute act_input and gelu_out in the bwd
        """
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight1, bias1, weight2, bias2 = [
                a.to(dtype=dtype) for a in [x, weight1, bias1, weight2, bias2]
            ]
        is_bf16 = x.dtype == torch.bfloat16
        assert checkpoint_lvl in [0, 1, 2]
        x = x.contiguous()
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous()
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        if is_bf16:
            act_input = fused_dense_cuda.linear_bias_forward(
                x.reshape(batch_dim, n), weight1, bias1
            )
            output1 = sqrelu_fwd(act_input)
        else:
            save_act_input = checkpoint_lvl != 2
            result = triton_linear_act(
                x.reshape(batch_dim, n),
                weight1,
                bias1,
                activation="squared_relu",
                save_act_input=save_act_input,
            )
            if save_act_input:
                output1, act_input = result
            else:
                output1 = result
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, bias1, weight2, act_input, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, bias1, weight2, act_input)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, bias1, weight2)
        return output2.reshape(*batch_shape, output2.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        x, weight1, bias1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        is_bf16 = x.dtype == torch.bfloat16
        if checkpoint_lvl == 0:
            act_input, output1 = rest
        elif checkpoint_lvl == 1:
            (act_input,) = rest
            output1 = sqrelu_fwd(act_input)
        elif checkpoint_lvl == 2:
            if is_bf16:
                act_input = fused_dense_cuda.linear_bias_forward(
                    x.reshape(batch_dim, n), weight1, bias1
                )
                output1 = sqrelu_fwd(act_input)
            else:
                output1, act_input = triton_linear_act(
                    x.reshape(batch_dim, n),
                    weight1,
                    bias1,
                    activation="squared_relu",
                    save_act_input=True,
                )

        if is_bf16:
            grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
            grad_output1 = grad_output @ weight2
            grad_act_input = sqrelu_bwd(grad_output1, act_input)
            grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
                x.reshape(batch_dim, n), weight1, grad_act_input
            )
        else:
            grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
            grad_act_input = triton_dgrad_act(
                grad_output, weight2, activation="squared_relu", act_input=act_input
            )
            grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
                x.reshape(batch_dim, n), weight1, grad_act_input
            )
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


fused_dense_sqrelu_dense_function = FusedDenseSqreluDenseFunc.apply


class FusedDenseSqreluDense(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=True,
        bias2=True,
        checkpoint_lvl=0,
        device=None,
        dtype=None,
    ):
        """
        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        assert bias1 == True, "DenseSqreluDense module without bias is currently not supported"
        assert bias2 == True, "DenseSqreluDense module without bias is currently not supported"
        self.checkpoint_lvl = checkpoint_lvl
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        assert x.is_cuda
        return fused_dense_sqrelu_dense_function(
            x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.checkpoint_lvl
        )
