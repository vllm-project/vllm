# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
    str_dtype_to_torch_dtype,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
    scaled_quantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import MixtureOfExperts, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _mimo_v2_qkv_pair_key(name: str) -> tuple[str, str] | None:
    if name.endswith(".qkv_proj.weight"):
        return name[: -len(".weight")], "weight"
    if name.endswith(".qkv_proj.weight_scale_inv"):
        return name[: -len(".weight_scale_inv")], "scale"
    return None


def _mimo_v2_kv_shard_range(
    total_num_kv_heads: int,
    head_size: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[int, int]:
    if tp_size >= total_num_kv_heads:
        num_replicas = divide(tp_size, total_num_kv_heads)
        kv_head_idx = tp_rank // num_replicas
        return kv_head_idx * head_size, head_size

    num_heads = divide(total_num_kv_heads, tp_size)
    shard_size = num_heads * head_size
    return tp_rank * shard_size, shard_size


def _mimo_v2_qkv_dims(
    config,
    layer_idx: int | None,
) -> tuple[int, int, int, int, int, int, int]:
    is_swa = (
        layer_idx is not None
        and hasattr(config, "hybrid_layer_pattern")
        and layer_idx < len(config.hybrid_layer_pattern)
        and config.hybrid_layer_pattern[layer_idx] == 1
    )
    if is_swa:
        head_dim = getattr(config, "swa_head_dim", config.head_dim)
        v_head_dim = getattr(config, "swa_v_head_dim", config.v_head_dim)
        num_heads = getattr(
            config, "swa_num_attention_heads", config.num_attention_heads
        )
        num_kv_heads = getattr(
            config, "swa_num_key_value_heads", config.num_key_value_heads
        )
    else:
        head_dim = config.head_dim
        v_head_dim = getattr(config, "v_head_dim", head_dim)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

    q_rows = num_heads * head_dim
    k_rows = num_kv_heads * head_dim
    v_rows = num_kv_heads * v_head_dim
    return q_rows, k_rows, v_rows, head_dim, v_head_dim, num_heads, num_kv_heads


def _mimo_v2_dequant_block_shard(
    weight: torch.Tensor,
    scale: torch.Tensor,
    row_start: int,
    row_count: int,
    block_n: int,
    block_k: int,
    device: torch.device,
) -> torch.Tensor:
    block_start = row_start // block_n
    block_end = cdiv(row_start + row_count, block_n)
    expanded_start = block_start * block_n
    expanded_rows = (block_end - block_start) * block_n
    available_rows = min(expanded_rows, weight.shape[0] - expanded_start)
    expanded_weight = weight.narrow(0, expanded_start, available_rows).to(device)
    if available_rows != expanded_rows:
        padded = torch.empty(
            (expanded_rows, weight.shape[1]),
            dtype=expanded_weight.dtype,
            device=device,
        )
        padded.zero_()
        padded[:available_rows] = expanded_weight
        expanded_weight = padded
    expanded_scale = scale.narrow(0, block_start, block_end - block_start).to(device)
    dequant_weight = scaled_dequantize(
        expanded_weight,
        expanded_scale,
        group_shape=GroupShape(block_n, block_k),
        out_dtype=torch.float32,
    )
    local_start = row_start - expanded_start
    return dequant_weight.narrow(0, local_start, row_count)


def _mimo_v2_quantize_block_weight(
    weight: torch.Tensor,
    block_n: int,
    block_k: int,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    padded_rows = round_up(rows, block_n)
    padded_cols = round_up(cols, block_k)
    if padded_rows != rows or padded_cols != cols:
        padded = torch.empty(
            (padded_rows, padded_cols),
            dtype=weight.dtype,
            device=weight.device,
        )
        padded.zero_()
        padded[:rows, :cols] = weight
        weight = padded

    qweight, scale = scaled_quantize(
        weight,
        GroupShape(block_n, block_k),
        quant_dtype=quant_dtype,
        compute_dtype=torch.float32,
    )
    return qweight[:rows, :cols].contiguous(), scale


def _mimo_v2_copy_paired_qkv_fp8(
    *,
    config,
    weight_name: str,
    scale_name: str,
    weight_param: torch.nn.Parameter,
    scale_param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    block_size: list[int],
) -> None:
    block_n, block_k = block_size
    layer_idx = extract_layer_index(weight_name)
    (
        q_rows,
        k_rows,
        v_rows,
        head_dim,
        v_head_dim,
        num_heads,
        num_kv_heads,
    ) = _mimo_v2_qkv_dims(config, layer_idx)
    expected_rows = q_rows + k_rows + v_rows
    if loaded_weight.shape[0] != expected_rows:
        raise ValueError(
            f"{weight_name} has {loaded_weight.shape[0]} rows, expected "
            f"{expected_rows} from q={q_rows}, k={k_rows}, v={v_rows}."
        )

    ckpt_tp = num_kv_heads
    q_heads_per_ckpt_rank = divide(num_heads, ckpt_tp)
    ckpt_q_rows = q_heads_per_ckpt_rank * head_dim
    ckpt_k_rows = head_dim
    ckpt_v_rows = v_head_dim
    ckpt_chunk_rows = ckpt_q_rows + ckpt_k_rows + ckpt_v_rows
    ckpt_chunk_scale_rows = cdiv(ckpt_chunk_rows, block_n)
    if (
        loaded_weight.shape[0] == ckpt_tp * ckpt_chunk_rows
        and loaded_scale.shape[0] == ckpt_tp * ckpt_chunk_scale_rows
    ):
        logger.info_once(
            "Detected MiMo-V2 TP%d pre-sharded fused-QKV FP8 checkpoint layout.",
            ckpt_tp,
        )
        if tp_size == ckpt_tp:
            local_weight = loaded_weight.narrow(
                0, tp_rank * ckpt_chunk_rows, ckpt_chunk_rows
            )
            local_scale = loaded_scale.narrow(
                0, tp_rank * ckpt_chunk_scale_rows, ckpt_chunk_scale_rows
            )
        else:
            device = weight_param.device

            def dequant_ckpt_shard(
                ckpt_rank: int,
                row_start: int,
                row_count: int,
            ) -> torch.Tensor:
                chunk_weight = loaded_weight.narrow(
                    0, ckpt_rank * ckpt_chunk_rows, ckpt_chunk_rows
                )
                chunk_scale = loaded_scale.narrow(
                    0, ckpt_rank * ckpt_chunk_scale_rows, ckpt_chunk_scale_rows
                )
                return _mimo_v2_dequant_block_shard(
                    chunk_weight,
                    chunk_scale,
                    row_start,
                    row_count,
                    block_n,
                    block_k,
                    device,
                )

            q_heads_per_rank = divide(num_heads, tp_size)
            q_head_start = tp_rank * q_heads_per_rank
            q_head_end = q_head_start + q_heads_per_rank
            q_parts: list[torch.Tensor] = []
            next_q_head = q_head_start
            while next_q_head < q_head_end:
                ckpt_rank = next_q_head // q_heads_per_ckpt_rank
                ckpt_head_start = ckpt_rank * q_heads_per_ckpt_rank
                part_head_end = min(q_head_end, ckpt_head_start + q_heads_per_ckpt_rank)
                part_rows = (part_head_end - next_q_head) * head_dim
                part_start = (next_q_head - ckpt_head_start) * head_dim
                q_parts.append(dequant_ckpt_shard(ckpt_rank, part_start, part_rows))
                next_q_head = part_head_end

            if tp_size >= num_kv_heads:
                num_replicas = divide(tp_size, num_kv_heads)
                kv_head_start = tp_rank // num_replicas
                kv_head_count = 1
            else:
                kv_head_count = divide(num_kv_heads, tp_size)
                kv_head_start = tp_rank * kv_head_count
            kv_head_end = kv_head_start + kv_head_count
            k_parts = [
                dequant_ckpt_shard(ckpt_rank, ckpt_q_rows, ckpt_k_rows)
                for ckpt_rank in range(kv_head_start, kv_head_end)
            ]
            v_parts = [
                dequant_ckpt_shard(ckpt_rank, ckpt_q_rows + ckpt_k_rows, ckpt_v_rows)
                for ckpt_rank in range(kv_head_start, kv_head_end)
            ]
            local_dense = torch.cat([*q_parts, *k_parts, *v_parts], dim=0)
            local_weight, local_scale = _mimo_v2_quantize_block_weight(
                local_dense,
                block_n,
                block_k,
                weight_param.dtype,
            )

        if tuple(local_weight.shape) != tuple(weight_param.shape):
            raise ValueError(
                f"{weight_name} local shard has shape "
                f"{tuple(local_weight.shape)}, expected "
                f"{tuple(weight_param.shape)}."
            )
        if tuple(local_scale.shape) != tuple(scale_param.shape):
            raise ValueError(
                f"{scale_name} local shard has shape "
                f"{tuple(local_scale.shape)}, expected "
                f"{tuple(scale_param.shape)}."
            )

        weight_param.data.copy_(local_weight.to(weight_param.device))
        scale_param.data.copy_(local_scale.to(scale_param.device))
        return

    q_scale_rows = cdiv(q_rows, block_n)
    k_scale_rows = cdiv(k_rows, block_n)
    v_scale_rows = cdiv(v_rows, block_n)
    expected_scale_rows = q_scale_rows + k_scale_rows + v_scale_rows
    if loaded_scale.shape[0] < expected_scale_rows:
        raise ValueError(
            f"{scale_name} has {loaded_scale.shape[0]} rows, expected at "
            f"least {expected_scale_rows}."
        )
    extra_scale_rows = loaded_scale.shape[0] - expected_scale_rows
    if extra_scale_rows:
        logger.info_once(
            "Dropping %d extra MiMo-V2 fused-QKV FP8 scale rows from %s. "
            "The checkpoint has q/k/v scale rows %d/%d/%d plus extras, "
            "while q/k/v weight rows imply %d/%d/%d.",
            extra_scale_rows,
            scale_name,
            q_scale_rows,
            k_scale_rows,
            v_scale_rows,
            q_rows,
            k_rows,
            v_rows,
        )

    q_weight = loaded_weight.narrow(0, 0, q_rows)
    k_weight = loaded_weight.narrow(0, q_rows, k_rows)
    v_weight = loaded_weight.narrow(0, q_rows + k_rows, v_rows)
    q_scale = loaded_scale.narrow(0, 0, q_scale_rows)
    k_scale = loaded_scale.narrow(0, q_scale_rows, k_scale_rows)
    v_scale = loaded_scale.narrow(0, q_scale_rows + k_scale_rows, v_scale_rows)

    q_shard_rows = divide(q_rows, tp_size)
    q_start = tp_rank * q_shard_rows
    k_start, k_shard_rows = _mimo_v2_kv_shard_range(
        config.num_key_value_heads, head_dim, tp_rank, tp_size
    )
    v_start, v_shard_rows = _mimo_v2_kv_shard_range(
        config.num_key_value_heads, v_head_dim, tp_rank, tp_size
    )

    direct_copy = all(
        value % block_n == 0
        for value in (
            q_start,
            q_shard_rows,
            k_start,
            k_shard_rows,
            v_start,
            v_shard_rows,
        )
    )

    if direct_copy:
        local_weight = torch.cat(
            [
                q_weight.narrow(0, q_start, q_shard_rows),
                k_weight.narrow(0, k_start, k_shard_rows),
                v_weight.narrow(0, v_start, v_shard_rows),
            ],
            dim=0,
        )
        local_scale = torch.cat(
            [
                q_scale.narrow(0, q_start // block_n, q_shard_rows // block_n),
                k_scale.narrow(0, k_start // block_n, k_shard_rows // block_n),
                v_scale.narrow(0, v_start // block_n, v_shard_rows // block_n),
            ],
            dim=0,
        )
    else:
        device = weight_param.device
        local_dense = torch.cat(
            [
                _mimo_v2_dequant_block_shard(
                    q_weight, q_scale, q_start, q_shard_rows, block_n, block_k, device
                ),
                _mimo_v2_dequant_block_shard(
                    k_weight, k_scale, k_start, k_shard_rows, block_n, block_k, device
                ),
                _mimo_v2_dequant_block_shard(
                    v_weight, v_scale, v_start, v_shard_rows, block_n, block_k, device
                ),
            ],
            dim=0,
        )
        local_weight, local_scale = _mimo_v2_quantize_block_weight(
            local_dense,
            block_n,
            block_k,
            weight_param.dtype,
        )

    if tuple(local_weight.shape) != tuple(weight_param.shape):
        raise ValueError(
            f"{weight_name} local shard has shape {tuple(local_weight.shape)}, "
            f"expected {tuple(weight_param.shape)}."
        )
    if tuple(local_scale.shape) != tuple(scale_param.shape):
        raise ValueError(
            f"{scale_name} local shard has shape {tuple(local_scale.shape)}, "
            f"expected {tuple(scale_param.shape)}."
        )

    weight_param.data.copy_(local_weight.to(weight_param.device))
    scale_param.data.copy_(local_scale.to(scale_param.device))


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiMoV2MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        dtype = getattr(config, "moe_router_dtype", "float32")
        self.gate_dtype = str_dtype_to_torch_dtype(dtype)
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=self.gate_dtype,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=self.gate_dtype)
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            router_logits_dtype=self.gate_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "MiMoV2MoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.gate_dtype is not None:
            gate_input = hidden_states.to(self.gate_dtype)
        else:
            gate_input = hidden_states
        router_logits = self.gate(gate_input)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        v_scale: float | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim

        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = v_scale
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=self.v_head_dim,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config if "mtp.layers" not in prefix else None,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > -1 else None

        # Use DiffKV backend when V has a different head dim than K.
        # Auto-pick FA-DiffKV when FA3/4 is usable on this device, else fall
        # back to TRITON_ATTN_DIFFKV.  Users can force a choice via
        # `--attention-backend <FLASH_ATTN_DIFFKV|TRITON_ATTN_DIFFKV>`.
        if self.v_head_dim != self.head_dim:
            requested = get_current_vllm_config().attention_config.backend
            if requested is not None and requested.name.endswith("_DIFFKV"):
                backend_enum = requested
            else:
                fa_backend = AttentionBackendEnum.FLASH_ATTN_DIFFKV.get_class()
                if fa_backend.is_supported_on_current_device(
                    head_size=self.head_dim,
                    head_size_v=self.v_head_dim,
                    has_sinks=self.attention_sink_bias is not None,
                ):
                    backend_enum = AttentionBackendEnum.FLASH_ATTN_DIFFKV
                else:
                    backend_enum = AttentionBackendEnum.TRITON_ATTN_DIFFKV
            attn_backend = backend_enum.get_class()
            attn_backend.set_head_size_v(self.v_head_dim)
            logger.info_once("Using %s for attention.", attn_backend.get_name())
        else:
            attn_backend = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
            attn_backend=attn_backend,
            head_size_v=self.v_head_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Apply v_scale before attention
        if self.v_scale is not None:
            v = v * self.v_scale

        attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        v_scale = getattr(config, "attention_value_scale", None)

        if self.is_compressed_softmax_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        if self.is_layer_sparse:
            self.mlp = MiMoV2MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_compressed_softmax_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_id] == 1


@support_torch_compile
class MiMoV2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = eplb_config.num_redundant_experts

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiMoV2FlashDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        qkv_buffers: dict[str, dict[str, torch.Tensor]] = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "mtp" in name:
                continue

            expert_matched = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                name_rewritten = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if (
                    name_rewritten.endswith(".bias") or name_rewritten.endswith("_bias")
                ) and name_rewritten not in params_dict:
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = param.weight_loader

                weight_loader(
                    param,
                    loaded_weight,
                    name_rewritten,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_rewritten)
                expert_matched = True
                break

            if expert_matched:
                continue
            qkv_pair = _mimo_v2_qkv_pair_key(name)
            if qkv_pair is not None:
                qkv_base_name, qkv_kind = qkv_pair
                weight_name = f"{qkv_base_name}.weight"
                scale_name = f"{qkv_base_name}.weight_scale_inv"
                has_paired_qkv = (
                    weight_name in params_dict
                    and scale_name in params_dict
                    and getattr(self.quant_config, "weight_block_size", None)
                    is not None
                )
                if not has_paired_qkv:
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                    continue

                qkv_buffer = qkv_buffers.setdefault(qkv_base_name, {})
                qkv_buffer[qkv_kind] = loaded_weight
                if "weight" in qkv_buffer and "scale" in qkv_buffer:
                    _mimo_v2_copy_paired_qkv_fp8(
                        config=self.config,
                        weight_name=weight_name,
                        scale_name=scale_name,
                        weight_param=params_dict[weight_name],
                        scale_param=params_dict[scale_name],
                        loaded_weight=qkv_buffer["weight"],
                        loaded_scale=qkv_buffer["scale"],
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        block_size=self.quant_config.weight_block_size,
                    )
                    loaded_params.add(weight_name)
                    loaded_params.add(scale_name)
                    del qkv_buffers[qkv_base_name]
                continue
            stacked_matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)

                if (
                    name_rewritten.endswith(".bias")
                    and name_rewritten not in params_dict
                ):
                    continue

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_rewritten)

                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            orig_name = name
            mapped_name = maybe_remap_kv_scale_name(name, params_dict)
            name = mapped_name if mapped_name is not None else orig_name

            if name not in params_dict:
                continue

            param = params_dict[name]

            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)

                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if qkv_buffers:
            missing = ", ".join(sorted(qkv_buffers))
            raise RuntimeError(
                "Missing fused-QKV FP8 weight/scale pair for MiMo-V2 "
                f"checkpoint tensors: {missing}"
            )

        return loaded_params


class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = MiMoV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
