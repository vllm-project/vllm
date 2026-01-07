# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .base import RotaryEmbedding


class FourierRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        # extra parameters for FoPE
        num_key_value_heads: int,
        num_inv_freq: int,
        fope_sep_head: bool,
        fope_init_factor: float,
    ):
        # fope related parameters
        self.num_key_value_heads = num_key_value_heads
        self.num_inv_freq = num_inv_freq
        self.fope_sep_head = fope_sep_head
        self.fope_init_factor = fope_init_factor

        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        # setup parameters
        self.input_dim = self.inv_freq.shape[-1]
        self.output_dim = self.inv_freq.shape[-1]
        self.cos_coef = nn.Parameter(
            torch.empty(num_key_value_heads, self.input_dim, self.output_dim),
            requires_grad=False,
        )
        self.sin_coef = nn.Parameter(
            torch.empty(num_key_value_heads, self.input_dim, self.output_dim),
            requires_grad=False,
        )
        self.cos_coef.weight_loader = self.fope_coef_weight_loader
        self.sin_coef.weight_loader = self.fope_coef_weight_loader

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )

        inv_freq_idx_selected = torch.ones_like(inv_freq, dtype=torch.bool)
        if self.num_inv_freq is not None:
            num_inv_freq = self.num_inv_freq
            inv_freq_idx_selected[num_inv_freq:] = False
        else:
            inv_freq_idx_selected = inv_freq > (
                2.0 * torch.pi / self.max_position_embeddings
            )
            num_inv_freq = inv_freq_idx_selected.sum().item()

        inv_freq = inv_freq[inv_freq_idx_selected]

        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        self.inv_freq = self._compute_inv_freq(self.base)
        # TODO: zhouxinyu, implement FoPE cos/sin cache computation
        return torch.zeros(1)

    def apply_rotary_emb(self):
        """Customized apply_rotary_emb function for FoPE."""
        pass

    def get_step_eye(self, _param):
        import math

        _step_eye = torch.zeros_like(_param)

        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i * step < self.input_dim:
                _step_eye[..., i * step, i] = 1.0

        return _step_eye

    def forward_native(self, x: torch.Tensor, positions: torch.Tensor):
        # expand x, positions to additional batch size dim
        if x.dim() == 2:
            # (seq_len, hidden_size) -> (bsz, seq_len, hidden_size)
            x = x.unsqueeze(0)
        if positions.dim() == 1:
            # (seq_len) -> (bsz, seq_len)
            positions = positions.unsqueeze(0)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
        )  # (40) -> (1, 40, 1) -> (bsz, 40, 1)
        position_ids_expanded = positions[
            :, None, :
        ].float()  # (bsz, seq_len) -> (bsz, 1, seq_len)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        batch_size, seq_len, hidden_size = x.shape
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            if self.fope_sep_head:
                pos_cos = (
                    freqs.cos()
                    .unsqueeze(1)
                    .expand(batch_size, self.num_key_value_heads, seq_len, -1)
                )
                pos_sin = (
                    freqs.sin()
                    .unsqueeze(1)
                    .expand(batch_size, self.num_key_value_heads, seq_len, -1)
                )
            else:
                pos_cos = freqs.cos()
                pos_sin = freqs.sin()

            if self.fope_sep_head:
                # (1, 1, 8192, 40) x (1, 40, 40) -> (1, 8192, 1, 40)
                sin = torch.einsum("bhtD, hDd -> bthd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("bhtD, hDd -> bthd", pos_cos, self.cos_coef.float())
            else:
                sin = torch.einsum("btD, Dd -> btd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("btD, Dd -> btd", pos_cos, self.cos_coef.float())

            sin = F.pad(
                input=sin,
                pad=(0, self.head_size // 2 - sin.size(-1)),
                mode="constant",
                value=1,
            )
            cos = F.pad(
                input=cos,
                pad=(0, self.head_size // 2 - cos.size(-1)),
                mode="constant",
                value=1,
            )

            sin = torch.cat((sin, sin), dim=-1)
            cos = torch.cat((cos, cos), dim=-1)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward_cuda(self, x: torch.Tensor, positions: torch.Tensor):
        # TODO, zhouxinyu, implement FoPE cuda forward computation
        return self.forward_native(x, positions)

    def fope_coef_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        num_key_value_heads = loaded_weight.size(0)

        if num_key_value_heads < world_size:
            n_replicate = world_size // num_key_value_heads
            world_size = num_key_value_heads
            rank = rank // n_replicate

        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
        param.copy_(loaded_weight)
