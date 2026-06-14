# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.cca_attn import CCAAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@CustomOp.register("cca")
class CCA(MambaBase, CustomOp):
    def __init__(
        self,
        config,
        cca_num_k_heads: int = 2,
        cca_num_q_heads: int = 8,
        hidden_size: int | None = None,
        head_dim: int = 128,
        cca_time0: int = 2,
        cca_time1: int = 2,
        layer_number: int = 0,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_number = layer_number
        self.prefix = prefix

        # Use the model's true hidden size unless explicitly overridden.
        # (In Megatron this is the lane's hidden_size_in.)
        self.hidden_size = int(hidden_size or config.hidden_size)

        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.padding0 = cca_time0 - 1
        self.padding1 = cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        self.num_k_heads = int(cca_num_k_heads)
        self.num_q_heads = int(cca_num_q_heads)

        # Geometry
        self.head_dim = int(head_dim)
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.sqrt_head_dim = np.sqrt(self.head_dim)
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        assert self.num_q_heads % self.num_k_heads == 0, (
            "q_heads must be a multiple of k_heads"
        )
        assert (self.latent_k_dim + self.latent_q_dim) == (
            self.num_k_heads + self.num_q_heads
        ) * self.head_dim

        # Projections
        self.linear_q = ReplicatedLinear(
            self.hidden_size,
            self.latent_q_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_q",
        )
        self.linear_k = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_k",
        )
        self.val_proj1 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.val_proj1",
        )
        self.val_proj2 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.val_proj2",
        )

        # Depthwise + grouped conv along sequence (exactly like Megatron)
        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.in_out_ch = in_out_ch
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time0,
                groups=in_out_ch,
                padding=0,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_k_heads + self.num_q_heads),
                padding=0,
                stride=1,
            ),
        )

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.cca(
            hidden_states,
            output,
            self.prefix,
        )

    def _rms_normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equivalent to RMSNorm with unit weights and eps=1e-12/head_dim.
        # Normalize one tensor at a time in fp32 to reduce peak memory versus
        # the custom rms_norm op, which materializes an additional fp32 output.
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)

        query_fp32 = query.to(torch.float32)
        q_norm = torch.linalg.vector_norm(query_fp32, ord=2, dim=-1, keepdim=True)
        query_fp32.mul_(torch.rsqrt(q_norm * q_norm + eps))
        query_fp32.mul_(sqrt_head_dim)
        query.copy_(query_fp32)

        key_fp32 = key.to(torch.float32)
        k_norm = torch.linalg.vector_norm(key_fp32, ord=2, dim=-1, keepdim=True)
        key_fp32.mul_(torch.rsqrt(k_norm * k_norm + eps))
        key_fp32.mul_(sqrt_head_dim)
        temp = self.temp.to(torch.float32).view(1, 1, self.num_k_heads, 1)
        if self.config.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_fp32.mul_(temp)
        key.copy_(key_fp32)
        return query, key

    def _add_grouped_qk_means_inplace(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_k_heads = key_base.shape[2]
        key_base_fp32 = key_base.float()
        query_pre_grouped = query_pre.view(
            *query_pre.shape[:2], num_k_heads, self.gqa_groups, query_pre.shape[-1]
        )
        query_out_grouped = query.view_as(query_pre_grouped)
        query_out_grouped.add_(query_pre_grouped, alpha=0.5)
        query_out_grouped.add_(key_base_fp32.unsqueeze(-2), alpha=0.5)

        query_pre_mean = torch.mean(query_pre_grouped, dim=-2, dtype=torch.float32)
        key.add_(query_pre_mean, alpha=0.5)
        key.add_(key_base_fp32, alpha=0.5)
        return query, key

    def _conv_qk_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Manual conv_qk for decode-sized inputs.

        Decode uses tiny sequence windows (currently total_padding + 1), so the
        generic conv path can spend a disproportionate amount of time on layout
        transforms and kernel setup. This manual implementation preserves the
        two-stage depthwise+grouped conv math while operating directly on the
        compact decode tensor.

        Input:  [N, C, S]
        Output: [N, C, S_out]
        """
        # Stage 1: depthwise conv over sequence.
        w0 = self.conv_qk[0].weight.squeeze(1)  # [C, K0]
        b0 = self.conv_qk[0].bias  # [C] or None

        x = x.to(w0.dtype)
        k0 = w0.shape[1]
        x_windows = x.unfold(-1, k0, 1)  # [N, C, L_mid, K0]
        mid = (x_windows * w0[:, None, :]).sum(dim=-1)  # [N, C, L_mid]
        if b0 is not None:
            mid = mid + b0[None, :, None]

        # Stage 2: grouped conv over the depthwise output.
        w1 = self.conv_qk[1].weight  # [C, D, K1]
        b1 = self.conv_qk[1].bias  # [C] or None
        g = self.num_k_heads + self.num_q_heads
        d = self.head_dim
        k1 = w1.shape[2]
        mid_windows = mid.view(mid.shape[0], g, d, mid.shape[-1]).unfold(-1, k1, 1)
        w1_grouped = w1.view(g, d, d, k1)
        out = torch.einsum("godk,sgdtk->sgot", w1_grouped, mid_windows)
        if b1 is not None:
            out = out + b1.view(1, g, d, 1)
        return out.reshape(x.shape[0], g * d, out.shape[-1])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()

        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, CCAAttentionMetadata)
            conv_states = self.kv_cache[0]
            prev_hs = self.kv_cache[1]
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            if state_indices_tensor_d is not None and state_indices_tensor_d.dim() > 1:
                state_indices_tensor_d = state_indices_tensor_d[:, 0]
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p

        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states.unsqueeze(0).transpose(0, 1).contiguous()
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))  # [S, B, H]
            q = self.linear_q(hs)  # [S, B, latent_q_dim]
            k = self.linear_k(hs)  # [S, B, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
            del q
            del k

            # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
            query_pre = qk_packed0[..., : self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )  # [S, B, qh, dh]

            key_base = qk_packed0[..., self.latent_q_dim :].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )  # [S, B, kh, dh]

            qk_packed1 = qk_packed0.permute(1, 2, 0)  # [B, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            # Build queries/keys from conv output + means
            query = (
                qk_packed3[..., : self.latent_q_dim]
                .view(*qk_packed3.shape[:2], self.num_q_heads, self.head_dim)
                .float()
            )

            key = (
                qk_packed3[..., self.latent_q_dim :]
                .view(*qk_packed3.shape[:2], self.num_k_heads, self.head_dim)
                .float()
            )
            query, key = self._add_grouped_qk_means_inplace(
                query, key, query_pre, key_base
            )
            del query_pre
            del key_base
            del qk_packed0
            del qk_packed3

            # Values from the two time streams
            v1 = self.val_proj1(hs)  # [S, B, latent_k_dim/2]
            v2 = self.val_proj2(hs_d)  # [S, B, latent_k_dim/2]
            value = (
                torch.cat([v1, v2], dim=-1)
                .contiguous()
                .view(*hs.shape[:2], self.num_k_heads, self.head_dim)
            )  # [S, B, kh, dh]

            query, key = self._rms_normalize_qk(query.contiguous(), key.contiguous())

            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states[:num_actual_tokens]

        # Batch size is effectively 1 in this path, so insert the singleton
        # dimension directly instead of transposing and materializing a copy.
        hs = hidden_states.unsqueeze(1)  # [S, 1, H]
        batch_size = hs.shape[1]

        q = self.linear_q(hs)  # [S, B, latent_q_dim]
        k = self.linear_k(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
        del q
        del k

        # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
        query_pre = qk_packed0[..., : self.latent_q_dim].view(
            *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
        )  # [S, B, qh, dh]

        key_base = qk_packed0[..., self.latent_q_dim :].view(
            *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        qk_packed0_d, qk_packed0_p = torch.split(
            qk_packed0[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        hs_d, hs_p = torch.split(
            hs[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        qk_packed3 = torch.empty(
            (num_actual_tokens, batch_size, self.in_out_ch),
            device=hs.device,
            dtype=hs.dtype,
        )
        hs2 = torch.empty(
            (num_actual_tokens, batch_size, self.hidden_size),
            device=hs.device,
            dtype=hs.dtype,
        )
        decode_is_pad: torch.Tensor | None = None
        if has_prefill:
            assert state_indices_tensor_p is not None
            assert has_initial_states_p is not None
            assert query_start_loc_p is not None
            # Prefill
            prefill_slice = slice(num_decodes, num_decodes + num_prefill_tokens)
            hs2_prefill = hs2[prefill_slice]
            qk_packed3_prefill = qk_packed3[prefill_slice]
            for i in range(len(query_start_loc_p) - 1):
                start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
                hs2_cur = hs_p[start_i:end_i, :, :]  # [S_cur, B, H]
                qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]  # [S_cur, B, H]
                qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)  # [1, H, S_cur]

                if has_initial_states_p[i]:
                    hs2_cached = (
                        prev_hs[state_indices_tensor_p[i]].unsqueeze(0).unsqueeze(0)
                    )  # [1, 1, H]
                    if hs2_cached.dtype != hs2_cur.dtype:
                        hs2_cached = hs2_cached.to(hs2_cur.dtype)
                    hs2_cur = torch.cat(
                        [hs2_cached, hs2_cur[:-1]], dim=0
                    )  # [S_cur, 1, H]
                    qk_packed0_cached = conv_states[
                        state_indices_tensor_p[i]
                    ].unsqueeze(0)  # [1, H, total_padding]
                    if qk_packed0_cached.dtype != qk_packed1_cur.dtype:
                        qk_packed0_cached = qk_packed0_cached.to(qk_packed1_cur.dtype)
                    qk_packed2_cur = torch.cat(
                        [qk_packed0_cached, qk_packed1_cur], dim=-1
                    )  # [1, H, S_cur + total_padding]
                else:
                    hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 0, 0, 1, 0))
                    qk_packed2_cur = F.pad(qk_packed1_cur, (self.total_padding, 0))

                hs2_prefill[start_i:end_i] = hs2_cur

                conv_states_cur = nn.functional.pad(
                    qk_packed2_cur, (self.cca_time0 - qk_packed2_cur.shape[-1], 0)
                )
                conv_states[state_indices_tensor_p[i]] = conv_states_cur.to(
                    device=conv_states.device, dtype=conv_states.dtype
                )

                # Computing conv
                qk_packed3_cur = self.conv_qk(qk_packed2_cur).permute(
                    2, 0, 1
                )  # [S, B, E]
                qk_packed3_prefill[start_i:end_i] = qk_packed3_cur

            prev_hs[state_indices_tensor_p] = hs_p[query_start_loc_p[1:] - 1, 0, :].to(
                device=prev_hs.device, dtype=prev_hs.dtype
            )

        if has_decode:
            assert state_indices_tensor_d is not None
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]
            decode_is_pad = state_indices_tensor_d == PAD_SLOT_ID
            # block_id=0 reserved
            # Zvllm/vllm/v1/core/block_pool.py
            safe_decode_indices = torch.where(
                decode_is_pad,
                torch.zeros_like(state_indices_tensor_d),
                state_indices_tensor_d,
            )
            qk_packed0_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_d.new_zeros(()),
                qk_packed0_d,
            )
            hs_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs_d.new_zeros(()),
                hs_d,
            )

            qk_packed0_cached = conv_states[
                safe_decode_indices
            ]  # [S, H, total_padding]
            qk_packed0_cached = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_cached.new_zeros(()),
                qk_packed0_cached,
            )
            qk_packed0_cached_for_compute = qk_packed0_cached
            decode_qk_dtype = qk_packed0_d.dtype
            if qk_packed0_cached_for_compute.dtype != decode_qk_dtype:
                qk_packed0_cached_for_compute = qk_packed0_cached_for_compute.to(
                    decode_qk_dtype
                )
            qk_packed0_cat = torch.cat(
                [qk_packed0_cached_for_compute, qk_packed0_d.transpose(1, 2)], dim=-1
            )  # [S, H, total_padding + 1]
            qk_packed3_d = self._conv_qk_decode(qk_packed0_cat).transpose(
                1, 2
            )  # [S, 1, E]
            qk_packed3[:num_decodes] = qk_packed3_d

            new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
            new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :].to(
                new_qk_packed0_cache.dtype
            )
            new_qk_packed0_cache = torch.where(
                decode_is_pad.view(-1, 1, 1),
                new_qk_packed0_cache.new_zeros(()),
                new_qk_packed0_cache,
            )
            conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                device=conv_states.device, dtype=conv_states.dtype
            )

            hs2_decode = prev_hs[safe_decode_indices].unsqueeze(1)  # [S, 1, H]
            hs2_decode = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs2_decode.new_zeros(()),
                hs2_decode,
            )
            if hs2_decode.dtype != hs.dtype:
                hs2_decode = hs2_decode.to(hs.dtype)
            hs2[:num_decodes] = hs2_decode
            new_prev_hs = hs_d[:, 0, :].to(prev_hs.dtype)
            new_prev_hs = torch.where(
                decode_is_pad.view(-1, 1),
                new_prev_hs.new_zeros(()),
                new_prev_hs,
            )
            prev_hs[safe_decode_indices] = new_prev_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype
            )

        del qk_packed0_d
        del qk_packed0_p
        del hs_d
        del hs_p

        # Values from the two time streams
        v1 = self.val_proj1(hs)  # [S, B, latent_k_dim/2]
        v2 = self.val_proj2(hs2)
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]
        del hs2

        # Build queries/keys from conv output + means
        query = (
            qk_packed3[..., : self.latent_q_dim]
            .view(num_actual_tokens, batch_size, self.num_q_heads, self.head_dim)
            .float()
        )

        key = (
            qk_packed3[..., self.latent_q_dim :]
            .view(num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)
            .float()
        )
        query, key = self._add_grouped_qk_means_inplace(query, key, query_pre, key_base)
        del query_pre
        del key_base
        del qk_packed0
        del qk_packed3

        query, key = self._rms_normalize_qk(query.contiguous(), key.contiguous())
        # Flatten the singleton batch dimension without transpose/cat copies and
        # write directly into the preallocated output buffer.
        query = query.reshape(num_actual_tokens, self.latent_q_dim)
        key = key.reshape(num_actual_tokens, self.latent_k_dim)
        value = value.reshape(num_actual_tokens, self.latent_k_dim)
        q_end = self.latent_q_dim
        k_end = q_end + self.latent_k_dim
        output[:num_actual_tokens, :q_end] = query
        output[:num_actual_tokens, q_end:k_end] = key
        output[:num_actual_tokens, k_end:] = value
        if decode_is_pad is not None:
            decode_output = output[:num_decodes]
            output[:num_decodes] = torch.where(
                decode_is_pad.view(-1, 1),
                decode_output.new_zeros(()),
                decode_output,
            )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.cca_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            conv_kernel_size=self.total_padding,
            num_k_heads=self.num_k_heads,
            num_q_heads=self.num_q_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )

    @property
    def mamba_type(self) -> str:
        return "cca"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.cca_attn import CCAAttentionBackend

        return CCAAttentionBackend


def cca(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def cca_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cca",
    op_func=cca,
    mutates_args=["output"],
    fake_impl=cca_fake,
)
