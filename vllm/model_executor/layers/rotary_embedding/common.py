# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from importlib.util import find_spec

import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


# common functions
def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# yarn functions
# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _flashinfer_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Custom op wrapper for flashinfer's rotary embedding.

    This is an in-place operation that modifies query and key tensors directly.
    """
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def _flashinfer_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return


# Register flashinfer rotary embedding custom op
direct_register_custom_op(
    op_name="flashinfer_rotary_embedding",
    op_func=_flashinfer_rotary_embedding,
    mutates_args=["query", "key"],  # These tensors are modified in-place
    fake_impl=_flashinfer_rotary_embedding_fake,
)


@CustomOp.register("apply_rotary_emb")
class ApplyRotaryEmb(CustomOp):
    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super().__init__(enforce_enable)
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute

        self.apply_rotary_emb_flash_attn = None
        if find_spec("flash_attn") is not None:
            from flash_attn.ops.triton.rotary import apply_rotary

            self.apply_rotary_emb_flash_attn = apply_rotary

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size (optional), seq_len, num_heads, head_size]
            cos: [seq_len, head_size // 2]
            sin: [seq_len, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style.
            enable_fp32_compute: Temporarily convert x, cos, sin to FP32 dtype
                                 for higher accuracy.
        """
        origin_dtype = x.dtype
        if enable_fp32_compute:
            x = x.float()

        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output

    def forward_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        output = self.forward_static(
            x, cos, sin, self.is_neox_style, self.enable_fp32_compute
        )
        return output

    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        origin_dtype = x.dtype
        if self.enable_fp32_compute:
            x = x.float()
            cos = cos.float()
            sin = sin.float()

        origin_shape = x.shape
        if len(origin_shape) == 3:
            # x: [seq_len, num_heads, head_size]
            x = x.unsqueeze(0)

        """
        Arguments of apply_rotary_emb() in vllm_flash_attn:
            x: [batch_size, seq_len, nheads, headdim]
            cos, sin: [seqlen_rotary, rotary_dim / 2]
            interleaved: defalut as False (Neox-style).
            ...
        """
        interleaved = not self.is_neox_style
        output = apply_rotary_emb(x, cos, sin, interleaved)

        if len(origin_shape) == 3:
            output = output.squeeze(0)
        if self.enable_fp32_compute:
            output = output.to(origin_dtype)
        return output

    def forward_hip(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.apply_rotary_emb_flash_attn is not None:
            origin_dtype = x.dtype
            if self.enable_fp32_compute:
                x = x.float()
                cos = cos.float()
                sin = sin.float()

            origin_shape = x.shape
            if len(origin_shape) == 3:
                # x: [seq_len, num_heads, head_size]
                x = x.unsqueeze(0)

            """
            Arguments of apply_rotary() in flash_attn:
                x: [batch_size, seq_len, nheads, headdim]
                cos, sin: [seqlen_rotary, rotary_dim / 2]
                interleaved: defalut as False (Neox-style).
                ...
            """
            interleaved = not self.is_neox_style
            output = self.apply_rotary_emb_flash_attn(
                x, cos, sin, interleaved=interleaved
            ).type_as(x)

            if len(origin_shape) == 3:
                output = output.squeeze(0)
            if self.enable_fp32_compute:
                output = output.to(origin_dtype)
        else:
            # Falling back to PyTorch native implementation.
            output = self.forward_native(x, cos, sin)

        return output

    def forward_cpu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # TODO (bigPYJ1151): need to enable fused CPU ROPE here
        return self.forward_native(x, cos, sin)

    def extra_repr(self) -> str:
        s = f"is_neox_style={self.is_neox_style}"
        s += f"enable_fp32_compute={self.enable_fp32_compute}"
        return s
