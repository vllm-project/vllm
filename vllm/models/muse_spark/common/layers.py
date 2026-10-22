# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral Common MuseSpark1x layers and residual connections."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd


def calculate_gain_center(
    gain_center_type: str,
    residual_id: int,
    n_layers: int | None = None,
) -> float:
    """Gain center for the given type and residual ID (1-based indices)."""
    if gain_center_type == "one":
        return 1.0
    if gain_center_type == "zero":
        return 0.0
    if gain_center_type == "inverse_residual_index":
        return 1.0 / (2 * math.ceil(residual_id / 2))
    if gain_center_type in (
        "inverse_residual_index_v2",
        "inverse_residual_index_p1",
    ):
        return 1.0 / (residual_id + 1)
    if gain_center_type == "inverse_sqrt_residual_index":
        return 1.0 / math.sqrt(2 * math.ceil(residual_id / 2))
    if gain_center_type == "inverse_sqrt_layers":
        assert n_layers is not None, "n_layers must be provided for this option."
        return 1.0 / math.sqrt(n_layers)
    raise ValueError(f"Unsupported gain_center_type: {gain_center_type}")


class MuseSpark1xRMSNorm(nn.Module):
    """RMSNorm in fp32, rounded once at the store to ``out_dtype``.

    Deliberately not ``F.rms_norm``: that path can flip greedy token selection.
    """

    # Which side of the normalization the weight lands on; read by fused kernels.
    applies_weight_before_norm = False

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        is_post_norm: bool = False,
        zero_centered_gamma: bool = False,
        post_norm_gain_center_type: str = "one",
        residual_id: int | None = None,
        output_norm_gain_center_type: str = "zero",
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        self.has_weight = has_weight
        if self.has_weight:
            self.weight = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        else:
            self.weight = None

        self.zero_centered_gamma = zero_centered_gamma
        self.gain_center = 1.0
        if residual_id is not None and is_post_norm:
            self.gain_center = calculate_gain_center(
                post_norm_gain_center_type, residual_id
            )
        elif residual_id == -1:
            self.gain_center = calculate_gain_center(
                output_norm_gain_center_type, residual_id
            )

    @property
    def fused_gain_center(self) -> float:
        """``gain_center`` as the fused kernels take it: 0.0 when unused."""
        return self.gain_center if self.has_weight and self.zero_centered_gamma else 0.0

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        out_dtype = out_dtype or orig_dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )
            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            weight = self.weight.float()
            if self.zero_centered_gamma:
                weight = weight + self.gain_center
            x = x * weight
        x = x.to(out_dtype)
        if residual is None:
            return x
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None and self.supports_fused_kernel(x):
            return fused_rmsnorm_fwd(
                x,
                self.weight if self.has_weight else None,
                self.variance_epsilon,
                out_dtype=out_dtype,
                gain_center=self.fused_gain_center,
                input_scale=self.applies_weight_before_norm and self.has_weight,
                # Only matters past one 8192-element tile; a device query here
                # would break the compiled graph.
                use_kahan_summation=True,
            )
        return self.forward_native(x, residual, out_dtype)

    def supports_fused_kernel(self, x: torch.Tensor) -> bool:
        """Whether the Triton kernel can serve this norm for input ``x``."""
        return (
            self.variance_size_override is None
            and x.is_contiguous()
            and x.shape[-1] == self.hidden_size
        )


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    """Weight-first RMSNorm variant: ``rms_norm(x * weight)``."""

    applies_weight_before_norm = True

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        out_dtype = out_dtype or orig_dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        if self.weight is not None:
            weight = self.weight.float()
            if self.zero_centered_gamma:
                weight = weight + self.gain_center
            x = x * weight
        x = F.rms_norm(x, (self.hidden_size,), None, self.variance_epsilon)
        x = x.to(out_dtype)

        if residual is None:
            return x
        return x, residual


class MulTanhSoftCap(nn.Module):
    """Computes ``soft_cap_temp * tanh(x * multiplier / soft_cap_temp)`` in fp32."""

    def __init__(self, multiplier: float, soft_cap_temp: float) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.soft_cap_temp = soft_cap_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        if x.is_cuda and x.is_contiguous():
            if x.dtype != torch.float32:
                x = x.float()
            return ops.muse_spark_mul_tanh_soft_cap(
                x, self.multiplier, self.soft_cap_temp
            )
        x = x.float() * self.multiplier
        return self.soft_cap_temp * torch.tanh(x / self.soft_cap_temp)


class MuseSpark1xResidualConnection(nn.Module):
    """Vanilla residual sum: ``stream + branch``."""

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return stream + branch.to(stream.dtype)


class MuseSpark1xNormPreservingResidualConnection(MuseSpark1xResidualConnection):
    """Gated residual sum with norm-preserving weights: ``alpha*a + beta*b``.

    With ``a`` the stream and ``b`` the branch, and gate parameter ``w``:

        beta = sigmoid(w / temperature)
        alpha = sqrt(max(eps, sigmoid(-w) * (1 + beta)))   # = sqrt(1 - beta^2)

    The sum is computed in fp32 and cast back to the stream dtype.
    """

    def __init__(
        self,
        dim: int,
        activation: str = "sigmoid",
        temperature: float = 0.3,
        use_stream_gate: bool = True,
        channel_wise: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.eps = eps
        self.activation_type = activation
        self.temperature = temperature
        self.use_stream_gate = use_stream_gate
        self.gate = nn.Parameter(torch.empty(dim if channel_wise else 1))

    def coefficients(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor]:
        """Compute post-activation coefficients for the stream and branch."""
        if self.activation_type != "sigmoid":
            raise ValueError(f"Unsupported activation: {self.activation_type}")
        beta = torch.sigmoid(w)
        if self.use_stream_gate:
            alpha_sq = torch.sigmoid(-w) * (1.0 + beta)  # = 1 - beta^2 (stable)
            alpha = torch.sqrt(torch.clamp(alpha_sq, min=self.eps))
        else:
            alpha = 1.0
        return alpha, beta

    def materialize_coefficients(self) -> None:
        """Cache alpha/beta as fp32 buffers. Call once, after weight loading."""
        w = self.gate.detach().float() / self.temperature
        alpha, beta = self.coefficients(w)
        self.register_buffer("beta", beta.contiguous(), persistent=False)
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.contiguous(), persistent=False)
        else:
            self.alpha = alpha
        self._coefficients_ready = True

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        dtype = stream.dtype
        branch = branch.to(dtype)

        if getattr(self, "_coefficients_ready", False):
            alpha, beta = self.alpha, self.beta
        else:
            w = self.gate.view(*([1] * (stream.ndim - 1)), -1).float()
            alpha, beta = self.coefficients(w / self.temperature)
        out = alpha * stream + beta * branch
        return out.to(dtype)


@dataclass(slots=True)
class MuseSpark1xResidualStream:
    """A residual stream whose last blend is deferred to fuse with the next norm."""

    stream: torch.Tensor
    branch: torch.Tensor
    post_norm: MuseSpark1xRMSNorm
    connection: MuseSpark1xResidualConnection


def norm_then_gated_residual(x: MuseSpark1xResidualStream) -> torch.Tensor:
    """``alpha * stream + beta * norm(branch)``, in one kernel where possible."""
    norm, connection = x.post_norm, x.connection
    branch, stream = x.branch, x.stream
    if (
        isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and branch.is_cuda
        and stream.shape == branch.shape
        and stream.is_contiguous()
        and norm.supports_fused_kernel(branch)
    ):
        return fused_rmsnorm_fwd(
            branch,
            norm.weight if norm.has_weight else None,
            norm.variance_epsilon,
            gain_center=norm.fused_gain_center,
            input_scale=norm.applies_weight_before_norm and norm.has_weight,
            use_kahan_summation=True,
            residual_stream=stream,
            alpha=connection.alpha,
            beta=connection.beta,
        )
    return connection(branch=norm(branch), stream=stream)


def materialize_residual_path_constants(model: nn.Module) -> int:
    """Precompute parameter-only constants after weight loading; idempotent."""
    n_gates = 0
    for module in model.modules():
        if isinstance(module, MuseSpark1xNormPreservingResidualConnection):
            module.materialize_coefficients()
            n_gates += 1
        else:
            fold = getattr(module, "fold_router_output_multiplier", None)
            if callable(fold):
                fold()
    return n_gates


class MuseSpark1xQKVOGateParallelLinear(ColumnParallelLinear):
    """Fused [Q, K, V, O-gate] projection with head-aware TP sharding.

    Per-rank rows are [q, k, v, o_gate], so checkpoint tensors are sliced by
    head group. KV-head padding is not supported.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        self.head_size = head_size
        super().__init__(
            input_size=hidden_size,
            output_size=2 * (total_num_heads + total_num_kv_heads) * head_size,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        assert total_num_heads % self.tp_size == 0
        assert total_num_kv_heads % self.tp_size == 0, (
            "KV-head padding is not supported for the fused QKV+O-gate "
            f"projection: {total_num_kv_heads=}, tp_size={self.tp_size}."
        )
        self.num_heads = total_num_heads // self.tp_size
        self.num_kv_heads = total_num_kv_heads // self.tp_size

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        if loaded_shard_id is None:
            assert param.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            shard_offset, shard_size = 0, q_size
        elif loaded_shard_id == "k":
            shard_offset, shard_size = q_size, kv_size
        elif loaded_shard_id == "v":
            shard_offset, shard_size = q_size + kv_size, kv_size
        elif loaded_shard_id == "o_gate":
            shard_offset, shard_size = q_size + 2 * kv_size, q_size
        else:
            raise ValueError(
                f"{self.prefix}: unknown shard id {loaded_shard_id!r} for the "
                "fused QKV+O-gate projection."
            )
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_shard = param.data.narrow(0, shard_offset, shard_size)
        assert param_shard.shape == loaded_weight.shape
        param_shard.copy_(loaded_weight)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            if name == "bias" and self.bias is None:
                continue
            param = getattr(self, name)
            self.weight_loader(param, loaded_weight, shard_id)
            yield name


__all__ = [
    "MulTanhSoftCap",
    "MuseSpark1xNormPreservingResidualConnection",
    "MuseSpark1xQKVOGateParallelLinear",
    "MuseSpark1xRMSNorm",
    "MuseSpark1xRMSNormWithInputScale",
    "MuseSpark1xResidualConnection",
    "MuseSpark1xResidualStream",
    "calculate_gain_center",
    "materialize_residual_path_constants",
    "norm_then_gated_residual",
]
