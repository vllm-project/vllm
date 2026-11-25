# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.base import (
    Conv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.selector import (
    select_conv1d_decode_kernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.base import (
    Conv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.selector import (
    select_conv1d_prefill_kernel,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec


# --8<-- [start:short_conv]
@PluggableLayer.register("short_conv")
class ShortConv(MambaBase, PluggableLayer):
    # --8<-- [end:short_conv]

    conv_prefill_kernel: type[Conv1dPrefillKernel]
    conv_decode_kernel: type[Conv1dDecodeKernel]

    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_dim = dim
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        self._checkpoint_exporter = MambaPrefillCheckpointExporter(
            state_len=self.L_cache - 1
        )

        self.conv = ColumnParallelLinear(
            input_size=self.L_cache,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv.weight.data = self.conv.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        vllm_config = get_current_vllm_config()
        self.num_spec = vllm_config.num_speculative_tokens
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        spec: MambaSpec | None = None
        if model_config is not None:
            layer_config = self.get_kv_cache_config(vllm_config)
            assert layer_config is not None
            assert isinstance(layer_config.spec, MambaSpec)
            spec = layer_config.spec
        self.conv_prefill_kernel = select_conv1d_prefill_kernel(
            vllm_config, spec, width=self.L_cache, activation=None
        )
        self.conv_decode_kernel = select_conv1d_decode_kernel(
            vllm_config, spec, width=self.L_cache, activation=None
        )
        if spec is not None:
            dim_first = is_conv_state_dim_first()
            conv_dim, conv_state_len = spec.shapes[0][:: 1 if dim_first else -1]
            conv_bias = self.conv.bias
            self.conv_prefill_kernel.register_warmup(
                dim=conv_dim,
                state_len=conv_state_len,
                dim_first=dim_first,
                # The kernel convolves B * x, a new (tokens, dim) tensor.
                x_row_stride=conv_dim,
                weight_dtype=self.conv.weight.dtype,
                bias_dtype=None if conv_bias is None else conv_bias.dtype,
                max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.short_conv(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        attn_metadata: AttentionMetadata | None = None
        attn_metadata_raw = get_forward_context().attn_metadata
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw.get(self.prefix)

        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)
        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
            # Profile run
            output_tensor, _ = self.out_proj(C * (B * x))
            output[: hidden_states.shape[0]] = output_tensor
            return

        assert isinstance(attn_metadata, ShortConvAttentionMetadata)
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        num_decodes = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_actual_tokens = num_decodes + num_prefill_tokens

        # Decodes come before prefills.
        B_d, B_p = torch.split(B[:num_actual_tokens], [num_decodes, num_prefill_tokens])
        C_d, C_p = torch.split(C[:num_actual_tokens], [num_decodes, num_prefill_tokens])
        x_d, x_p = torch.split(x[:num_actual_tokens], [num_decodes, num_prefill_tokens])
        conv_output_list = []

        if attn_metadata.num_prefills > 0:
            Bx_p = (B_p * x_p).transpose(0, 1)
            Bx = self.conv_prefill_kernel.forward(
                Bx_p, conv_state, conv_weights, self.conv.bias, None, attn_metadata
            ).transpose(0, 1)[:num_prefill_tokens]
            if attn_metadata.checkpoint is not None:
                assert attn_metadata.query_start_loc_p is not None
                self._checkpoint_exporter.export(
                    attn_metadata.checkpoint,
                    raw_qkv=Bx_p.transpose(0, 1),
                    conv_state=conv_state,
                    cu_seqlens=attn_metadata.query_start_loc_p,
                )
            conv_output_list.append(C_p * Bx)

        if num_decodes > 0:
            Bx = self.conv_decode_kernel.forward(
                (B_d * x_d).contiguous(),
                conv_state,
                conv_weights,
                self.conv.bias,
                None,
                attn_metadata,
                max_query_len=1 + self.num_spec,
            )
            conv_output_list.insert(0, C_d * Bx)

        output[:num_actual_tokens], _ = self.out_proj(torch.vstack(conv_output_list))

    def get_kv_cache_checkpoint_alignment(self, vllm_config: VllmConfig) -> int | None:
        # The window ending at a checkpoint is a copy of its last inputs, so
        # any position works; the CPU path has no export.
        return None if current_platform.is_cpu() else 1

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.conv_dim,
            conv_kernel=self.L_cache,
            num_spec=self.num_spec,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV


@eager_break_during_capture
def short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(hidden_states=hidden_states, output=output)


direct_register_custom_op(
    op_name="short_conv",
    op_func=short_conv,
    mutates_args=["output"],
)
