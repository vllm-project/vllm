# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom activation functions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.collection_utils import LazyDict

logger = init_logger(__name__)


# --8<-- [start:fatrelu_and_mul]
@CustomOp.register("fatrelu_and_mul")
class FatreluAndMul(CustomOp):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:fatrelu_and_mul]

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.fatrelu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = F.threshold(x1, self.threshold, 0.0)
        return x1 * x2

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x, self.threshold)
        return out


# --8<-- [start:silu_and_mul]
@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:silu_and_mul]

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            self.op = ipex_ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out


# --8<-- [start:mul_and_silu]
@CustomOp.register("mul_and_silu")
class MulAndSilu(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:mul_and_silu]

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.mul_and_silu
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            self.op = ipex_ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return x[..., :d] * F.silu(x[..., d:])

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    # TODO implement forward_xpu for MulAndSilu
    # def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:


# --8<-- [start:gelu_and_mul_sparse]
@CustomOp.register("gelu_and_mul_sparse")
class GeluAndMulSparse(CustomOp):
    """An activation function for GeluAndMulSparse.
    This activation function is used in Gemma3n. It computes:
        up_proj = self.up_proj(x)
        gate_proj = self.gate_proj(x)
        gate_proj = self._gaussian_topk(gate_proj) # sparsity
        activations = self.act_fn(gate_proj) # gelu
        down_proj = self.down_proj(activations * up_proj)
    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:gelu_and_mul_sparse]

    def __init__(self, activation_sparsity: float, approximate: str = "none"):
        super().__init__()
        # Gelu.
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        if current_platform.is_rocm() and approximate == "tanh":
            # TODO:[ROCm] PyTorch native GELU with tanh is unstable with torch.compile
            logger.warning_once(
                "[ROCm] Pytorch's native GELU with tanh approximation is currently "
                "unstable and produces garbage. Fallback to 'none' approximation."
            )
            self.approximate = "none"

        # Sparsity.
        if activation_sparsity == 0.0:
            raise ValueError("activation_sparsity is 0.0. Please use GeluAndMul.")
        target_sparsity_tensor = torch.tensor(activation_sparsity, dtype=torch.float32)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        self.std_multiplier = normal_dist.icdf(target_sparsity_tensor)

    def _gaussian_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Get % sparse percentile of the Gaussian distribution."""
        # NOTE(rob): for TP>1, we could all-gather to get the means/std.
        # But we do not do this because in expectation they are the same
        # and in practice the eval scores are good without gathering.
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = mean + std * self.std_multiplier
        return nn.functional.relu(x - cutoff_x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        out = self._gaussian_topk(x[..., :d])
        out = F.gelu(out, approximate=self.approximate)
        return out * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


# --8<-- [start:gelu_and_mul]
@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    # --8<-- [end:gelu_and_mul]

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            if approximate == "none":
                self.op = torch.ops._C.gelu_and_mul
            elif approximate == "tanh":
                self.op = torch.ops._C.gelu_tanh_and_mul
        if current_platform.is_rocm() and approximate == "tanh":
            logger.warning_once(
                "[ROCm] PyTorch's native GELU with tanh approximation is unstable "
                "with torch.compile. For native implementation, fallback to 'none' "
                "approximation. The custom kernel implementation is unaffected."
            )
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            if approximate == "none":
                self.op = ipex_ops.gelu_and_mul
            else:
                self.op = ipex_ops.gelu_tanh_and_mul

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        # TODO: [ROCm] PyTorch's native GELU with tanh is unstable with torch.compile
        approximate = self.approximate
        if current_platform.is_rocm() and approximate == "tanh":
            approximate = "none"
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


# --8<-- [start:swigluoai_and_mul]
@CustomOp.register("swigluoai_and_mul")
class SwigluOAIAndMul(CustomOp):
    # https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L106-L110
    # --8<-- [end:swigluoai_and_mul]

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""

        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.swigluoai_and_mul(out, x, self.alpha, self.limit)
        return out

    def extra_repr(self) -> str:
        return f"alpha={repr(self.alpha)}, limit={repr(self.limit)}"


# --8<-- [start:gelu_new]
@CustomOp.register("gelu_new")
class NewGELU(CustomOp):
    # --8<-- [end:gelu_new]

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_new
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            self.op = ipex_ops.gelu_new

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# --8<-- [start:gelu_fast]
@CustomOp.register("gelu_fast")
class FastGELU(CustomOp):
    # --8<-- [end:gelu_fast]

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_fast
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            self.op = ipex_ops.gelu_fast

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# --8<-- [start:quick_gelu]
@CustomOp.register("quick_gelu")
class QuickGELU(CustomOp):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    # --8<-- [end:quick_gelu]

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_quick
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops

            self.op = ipex_ops.gelu_quick

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    # TODO implement forward_xpu for QuickGELU
    # def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:


# --8<-- [start:relu2]
@CustomOp.register("relu2")
class ReLUSquaredActivation(CustomOp):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    # --8<-- [end:relu2]

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return torch.square(F.relu(x))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO : implement cuda kernels
        return self.forward_native(x)


# --8<-- [start:xielu]
@CustomOp.register("xielu")
class XIELU(CustomOp):
    """
    Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010
    If the user has installed the nickjbrowning/XIELU, we import xIELU CUDA
    Otherwise, we emit a single warning and use xIELU Python
    """

    # --8<-- [end:xielu]

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
        with_vector_loads: bool = False,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1).unsqueeze(
                0
            )
        )
        self.alpha_n = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1
            ).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.with_vector_loads = with_vector_loads
        # Temporary until xIELU CUDA fully implemented
        self._beta_scalar = float(self.beta.detach().cpu().float().item())
        self._eps_scalar = float(self.eps.detach().cpu().float().item())

        self._xielu_cuda_obj = None
        try:
            import xielu.ops  # noqa: F401

            self._xielu_cuda_obj = torch.classes.xielu.XIELU()
            msg = "Using experimental xIELU CUDA."
            try:
                from torch._dynamo import allow_in_graph

                self._xielu_cuda_fn = allow_in_graph(self._xielu_cuda)
                msg += " Enabled torch._dynamo for xIELU CUDA."
            except Exception as err:
                msg += (
                    f" Could not enable torch._dynamo for xIELU ({err}) - "
                    "this may result in slower performance."
                )
                self._xielu_cuda_fn = self._xielu_cuda
            logger.warning_once(msg)
        except Exception as err:
            logger.warning_once(
                "CUDA-fused xIELU not available (%s) –"
                " falling back to a Python version.\n"
                "For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`",
                str(err),
            )

    def _xielu_python(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def _xielu_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Firewall function to prevent torch.compile from seeing .item()"""
        assert self._xielu_cuda_obj is not None, "XIELU CUDA object must not be None"
        original_shape = x.shape
        # CUDA kernel expects 3D tensors, reshape if needed
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.view(-1, 1, x.size(-1))
        if original_shape != x.shape:
            logger.warning_once(
                "Warning: xIELU input tensor expects 3 dimensions"
                " but got (shape: %s). Reshaping to (shape: %s).",
                original_shape,
                x.shape,
            )
        result = self._xielu_cuda_obj.forward(
            x,
            self.alpha_p,
            self.alpha_n,
            # Temporary until xIELU CUDA fully implemented ->
            # self.{beta,eps}.item()
            self._beta_scalar,
            self._eps_scalar,
            self.with_vector_loads,
        )
        return result.view(original_shape)

    def forward_native(self, input: torch.Tensor) -> torch.Tensor:
        if self._xielu_cuda_obj is not None and input.is_cuda:
            if not torch._dynamo.is_compiling():
                return self._xielu_cuda_fn(input)
            else:
                logger.warning_once(
                    "torch._dynamo is compiling, using Python version of xIELU."
                )
        return self._xielu_python(input)

    def forward_cuda(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_native(input)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = LazyDict(
    {
        "gelu": lambda: nn.GELU(),
        "gelu_fast": lambda: FastGELU(),
        "gelu_new": lambda: NewGELU(),
        "gelu_pytorch_tanh": lambda: (
            # TODO:[ROCm] PyTorch native GELU with tanh is unstable with torch.compile
            logger.warning_once(
                "[ROCm] PyTorch's native GELU with tanh approximation is unstable. "
                "Falling back to GELU(approximate='none')."
            ),
            nn.GELU(approximate="none"),
        )[1]
        if current_platform.is_rocm()
        else nn.GELU(approximate="tanh"),
        "relu": lambda: nn.ReLU(),
        "relu2": lambda: ReLUSquaredActivation(),
        "silu": lambda: nn.SiLU(),
        "quick_gelu": lambda: QuickGELU(),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "xielu": lambda: XIELU(),
    }
)


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()

    if act_fn_name.startswith("torch.nn.modules."):
        activation_name = act_fn_name.split(".")[-1]
        if activation_name == "identity":
            return nn.Identity()
        act_fn_name = activation_name

    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_REGISTRY[act_fn_name]


_ACTIVATION_AND_MUL_REGISTRY = LazyDict(
    {
        "gelu": lambda: GeluAndMul(),
        "silu": lambda: SiluAndMul(),
        "geglu": lambda: GeluAndMul(),
        "swigluoai": lambda *args, **kwargs: SwigluOAIAndMul(*args, **kwargs),
    }
)


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    """Get an activation-and-mul (i.e. SiluAndMul) function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]
