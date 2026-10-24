# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step5 model."""

import copy
import typing
from collections.abc import Iterable
from typing import Any

import regex as re
import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    KVTransferConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    skip_spec_layers,
)
from vllm.model_executor.parameter import BasevLLMParameter, ModelWeightParameter
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.step5.common.ops.attention_gate import headwise_sigmoid_gate
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend, AttentionType

from .config import (
    Step5SparseConfig,
    checkpoint_has_step5_sparse_config,
    get_step5_sparse_config,
)
from .layernorm import (
    Step5IndexerLayerNorm,
    Step5RMSNorm,
    get_step5_norm_dtype,
    make_step5_rms_norm,
)
from .ops.add import add_fp32

logger = init_logger(__name__)


def _can_fuse_decode_aux(metadata, runtime_mode: CUDAGraphMode) -> bool:
    # BCG prefill reuses captured projections/norms. Keep decode-only state
    # mutations out of those PIECEWISE segments. MRV2 captures FULL graphs
    # with runtime_mode=NONE, which is also the eager/warmup mode.
    return (
        runtime_mode != CUDAGraphMode.PIECEWISE
        and metadata is not None
        and metadata.num_actual_tokens > 0
        and metadata.max_query_len == 1
        and metadata.num_actual_tokens == metadata.num_actual_reqs
    )


def _is_uniform_verify(
    metadata, runtime_mode: CUDAGraphMode, num_speculative_tokens: int
) -> bool:
    return (
        runtime_mode != CUDAGraphMode.PIECEWISE
        and metadata is not None
        and metadata.num_actual_tokens > 0
        and metadata.max_query_len <= num_speculative_tokens + 1
        and metadata.num_actual_tokens
        == metadata.num_actual_reqs * metadata.max_query_len
    )


@eager_break_during_capture
@maybe_transfer_kv_layer
def step5_dsa_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    dsa_proxy_query: torch.Tensor,
    dsa_proxy_key: torch.Tensor,
    dsa_proxy_weights: torch.Tensor,
    dsa_proxy_z: torch.Tensor | None = None,
    dsa_summary_updated: bool = False,
    defer_inactive_clear: bool = False,
    gate: torch.Tensor | None = None,
) -> None:
    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    typing.cast(Any, attn_layer.impl).forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        dsa_proxy_query=dsa_proxy_query,
        dsa_proxy_key=dsa_proxy_key,
        dsa_proxy_weights=dsa_proxy_weights,
        dsa_proxy_z=dsa_proxy_z,
        dsa_summary_updated=dsa_summary_updated,
        defer_inactive_clear=defer_inactive_clear,
        dsa_gate=gate,
    )


@eager_break_during_capture
@maybe_transfer_kv_layer
def step5_swa_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    gate: torch.Tensor,
) -> None:
    metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    typing.cast(Any, layer.impl).forward_with_headwise_gate(
        layer, query, key, value, kv_cache, metadata, output, gate
    )


STEP5_SHARE_EXPERT_MAPPER = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"(?<!\.moe)\.share_expert\."): ".moe.share_expert.",
    },
)

STEP5_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
    "qkv_indexer_proj": [
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


# (packed parameter, checkpoint parameter, shard ID), in loading order.
_STEP5_WEIGHT_SHARDS: tuple[tuple[str, str, str | int], ...] = (
    ("qkvg_proj", "q_proj", 0),
    ("qkvg_proj", "k_proj", 1),
    ("qkvg_proj", "v_proj", 2),
    ("qkvg_proj", "g_proj", 3),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("qkv_indexer_proj", "q_proj", "q"),
    ("qkv_indexer_proj", "k_proj", "k"),
    ("qkv_indexer_proj", "v_proj", "v"),
    ("qkv_indexer_proj", "sparse_indexer_q", "index_q"),
    ("qkv_indexer_proj", "sparse_indexer_k", "index_k"),
    ("qkv_indexer_proj", "sparse_indexer_z", "index_z"),
    ("qkv_indexer_proj", "g_proj", "index_g"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _get_step5_layer_types(config: Any) -> list[str]:
    """Per-layer attention types spanning the dense stack and the MTP layers.

    `config.layer_types` only covers the dense stack so that transformers' own
    length validation passes; the MTP block indexes past it.
    """
    return (
        getattr(config, "layer_types_with_mtp", None)
        or getattr(config, "layer_types", None)
        or []
    )


def _parse_step5_layer_indices(
    value: str | Iterable[int] | None,
    *,
    name: str,
) -> set[int] | None:
    if value is None:
        return None
    raw_values = value.split(",") if isinstance(value, str) else value
    try:
        indices = [int(item) for item in raw_values if str(item).strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step5 {name} must contain integer layer indices.") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"Step5 {name} contains duplicate layer indices.")
    return set(indices)


def _get_step5_moe_layer_indices(config: Any) -> set[int]:
    enum_indices = _parse_step5_layer_indices(
        getattr(config, "moe_layers_enum", None),
        name="moe_layers_enum",
    )
    list_indices = _parse_step5_layer_indices(
        getattr(config, "moe_layer_list", None),
        name="moe_layer_list",
    )
    if (
        enum_indices is not None
        and list_indices is not None
        and enum_indices != list_indices
    ):
        raise ValueError(
            "Step5 moe_layers_enum and moe_layer_list describe different layers."
        )

    indices = enum_indices if enum_indices is not None else list_indices
    if indices is None:
        indices = set(range(1, int(config.num_hidden_layers)))
    total_layers = int(config.num_hidden_layers) + int(
        getattr(config, "num_nextn_predict_layers", 0) or 0
    )
    invalid = sorted(index for index in indices if not 0 <= index < total_layers)
    if invalid:
        raise ValueError(
            f"Step5 MoE layer indices must be in [0, {total_layers}), got {invalid}."
        )
    return indices


def _set_step5_moe_protocol_metadata(
    model: Any,
    example_layer: Any | None,
) -> None:
    """Populate the MoE protocol for the layers local to this PP rank."""
    model.num_moe_layers = len(model.moe_layers)
    model.num_expert_groups = 1
    model.num_shared_experts = 0
    if example_layer is None:
        # A valid pipeline stage can contain only dense layers. Reporting zero
        # local MoE layers keeps that rank out of EPLB while other stages still
        # expose their local expert runners.
        model.num_logical_experts = 0
        model.num_physical_experts = 0
        model.num_local_physical_experts = 0
        model.num_routed_experts = 0
        model.num_redundant_experts = 0
        return

    model.num_logical_experts = example_layer.n_logical_experts
    model.num_physical_experts = example_layer.n_physical_experts
    model.num_local_physical_experts = example_layer.n_local_physical_experts
    model.num_routed_experts = example_layer.n_routed_experts
    model.num_redundant_experts = example_layer.n_redundant_experts


def _verify_step5_dsa_kv_transfer_compatibility(vllm_config: VllmConfig) -> None:
    config = vllm_config.kv_transfer_config
    if config is None or not config.is_kv_transfer_instance:
        return
    configs = [config]
    if config.kv_connector == "MultiConnector":
        children = config.kv_connector_extra_config["connectors"]
        configs = [KVTransferConfig(**child) for child in children]
    if not all(
        c.kv_connector == "MooncakeStoreConnector"
        or (c.kv_connector == "NixlConnector" and c.kv_buffer_device == "cuda")
        for c in configs
    ):
        raise ValueError(
            "Step5 DSA sparse summaries are only transferred by "
            "MooncakeStoreConnector, or by NixlConnector with CUDA buffers, "
            "directly or under a MultiConnector."
        )


def _step5_groups_per_rank(total_groups: int, tp_size: int) -> int:
    """Return the number of contiguous groups visible on one TP rank."""
    if total_groups <= 0 or tp_size <= 0:
        raise ValueError(
            "Step5 parallel group counts must be positive, got "
            f"groups={total_groups}, tp_size={tp_size}."
        )
    if total_groups % tp_size == 0:
        return total_groups // tp_size
    if tp_size % total_groups == 0:
        return 1
    raise ValueError(
        f"Step5 tensor parallel size {tp_size} is incompatible with "
        f"group count {total_groups}: one must divide the other."
    )


def _step5_group_partition(
    *,
    total_groups: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[int, int, int]:
    """Return the first local group, group count, and replication factor."""
    if total_groups <= 0 or tp_size <= 0:
        raise ValueError(
            "Step5 replicated group counts must be positive, got "
            f"groups={total_groups}, tp_size={tp_size}."
        )
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"Step5 TP rank must be in [0, {tp_size}), got {tp_rank}.")
    local_groups = _step5_groups_per_rank(total_groups, tp_size)
    if tp_size <= total_groups:
        return tp_rank * local_groups, local_groups, 1
    ranks_per_group = tp_size // total_groups
    return tp_rank // ranks_per_group, 1, ranks_per_group


def _validate_step5_dsa_parallel_geometry(
    *,
    total_num_heads: int,
    total_num_kv_heads: int,
    indexer_num_heads: int,
    index_tp_size: int,
    tp_size: int,
) -> None:
    """Fail fast when DSA provider and attention GQA groups diverge.

    The sparse indexer reduces scores within provider groups, while the
    attention kernel consumes the corresponding KV groups.  These partitions
    must describe the same global and per-rank geometry; otherwise the model
    still constructs but selects regions for the wrong query/KV heads.
    """
    if total_num_heads <= 0 or total_num_kv_heads <= 0:
        raise ValueError(
            "Step5 attention head counts must be positive, got "
            f"num_heads={total_num_heads}, num_kv_heads={total_num_kv_heads}."
        )
    if index_tp_size <= 0:
        raise ValueError(
            "Step5 DSA provider group count must be positive, got "
            f"index_tp_size={index_tp_size}."
        )
    if index_tp_size != total_num_kv_heads:
        raise ValueError(
            "Step5 DSA provider groups must align with attention KV groups: "
            f"{index_tp_size} != {total_num_kv_heads}"
        )
    if indexer_num_heads <= 0 or indexer_num_heads % index_tp_size:
        raise ValueError(
            f"{indexer_num_heads} sparse indexer heads do not divide into "
            f"{index_tp_size} provider groups."
        )
    local_kv_groups = _step5_groups_per_rank(total_num_kv_heads, tp_size)
    local_provider_groups = _step5_groups_per_rank(index_tp_size, tp_size)
    if local_provider_groups != local_kv_groups:
        raise ValueError(
            "local DSA provider groups must align with local KV groups: "
            f"{local_provider_groups} != {local_kv_groups}"
        )
    if total_num_heads % tp_size:
        raise ValueError(
            "Step5 attention heads must be divisible by tensor parallel size: "
            f"num_heads={total_num_heads}, tp_size={tp_size}."
        )
    local_q_heads = total_num_heads // tp_size
    if local_q_heads % local_kv_groups:
        raise ValueError(
            f"{local_q_heads} local Q heads do not divide into "
            f"{local_kv_groups} local KV groups."
        )
    q_heads_per_kv = local_q_heads // local_kv_groups
    if q_heads_per_kv not in (4, 8, 16):
        raise ValueError(
            "Step5 DSA kernels require 4, 8, or 16 local query "
            "heads per KV group, got "
            f"{q_heads_per_kv} (num_heads={total_num_heads}, tp_size={tp_size})."
        )


def _per_layer_value(
    values: list[Any] | tuple[Any, ...] | None,
    layer_idx: int,
    *,
    name: str,
    default: Any,
) -> Any:
    if not values:
        return default
    if layer_idx >= len(values):
        raise ValueError(
            f"Step5 {name} has {len(values)} entries, but layer {layer_idx} "
            "requires an entry."
        )
    return values[layer_idx]


def _is_step5_full_attention_layer(
    config: Any,
    layer_idx: int,
    sparse_config: Any,
) -> bool:
    layer_types = _get_step5_layer_types(config)
    layer_type = (
        layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
    )
    apply_to = getattr(
        sparse_config,
        "apply_to_layer_types",
        ("full_attention",),
    )
    return layer_type in apply_to


class Step5FusedQKVIndexerLinear(QKVParallelLinear):
    """Main-attention qkv and the sparse-indexer q/k/z/g in one GEMM.

    Both projections consume the same normed hidden state, so they live in a
    single weight `[qkv_rows | indexer_rows]` and one `hidden @ W` replaces two
    GEMMs. The indexer rows are appended to the qkv rows of the base class, which
    keeps every q/k/v shard offset (and the kv-head replication for
    total_num_kv_heads < tp_size) inside `QKVParallelLinear.weight_loader_v2`
    instead of being reimplemented here.
    """

    INDEXER_SHARD_IDS = ("index_q", "index_k", "index_z", "index_g")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        indexer_q_output_size: int,
        indexer_kv_output_size: int,
        gate_output_size: int,
        proxy_dim: int,
        index_tp_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError(
                "Step5 DSA layers fuse qkv with the sparse indexer into one "
                "plain GEMM and cannot be quantized, but "
                f"{prefix}.weight resolved to "
                f"{type(self.quant_method).__name__}."
            )

        tp_size = self.tp_size
        self.proxy_dim = proxy_dim

        total_index_q_heads = indexer_q_output_size // self.proxy_dim
        total_index_kv_heads = indexer_kv_output_size // self.proxy_dim
        if tp_size >= total_index_kv_heads:
            self.local_index_kv_heads = 1
            self.index_kv_head_replicas = tp_size // total_index_kv_heads
        else:
            self.local_index_kv_heads = total_index_kv_heads // tp_size
            self.index_kv_head_replicas = 1
        # Keep each provider's complete index Q-head set. Providers are sharded
        # for TP <= index_tp_size and replicated above it, preserving top-k.
        self.index_group_start, self.local_index_groups, _ = _step5_group_partition(
            total_groups=index_tp_size,
            tp_size=tp_size,
            tp_rank=self.tp_rank,
        )
        if total_index_kv_heads <= 0:
            raise ValueError(
                "Step5 sparse indexer requires at least one KV head, got "
                f"{total_index_kv_heads}."
            )
        if total_index_kv_heads > tp_size:
            if total_index_kv_heads % tp_size != 0:
                raise ValueError(
                    "Step5 sparse indexer KV heads must divide the tensor "
                    "parallel size when sharded, got "
                    f"total_index_kv_heads={total_index_kv_heads}, "
                    f"tp_size={tp_size}."
                )
        elif tp_size % total_index_kv_heads != 0:
            raise ValueError(
                "Step5 sparse indexer KV heads must divide tensor parallel "
                "size when replicated, got "
                f"total_index_kv_heads={total_index_kv_heads}, tp_size={tp_size}."
            )
        self.local_index_q_heads = (
            total_index_q_heads // index_tp_size * self.local_index_groups
        )
        self.local_gate_size = gate_output_size // tp_size

        self.qkv_size = self.output_size_per_partition
        self.indexer_size = (
            self.local_index_q_size
            + self.local_index_k_size
            + self.local_index_z_size
            + self.local_gate_size
        )
        # Replace QKV-only storage before loading to include the indexer rows.
        weight = ModelWeightParameter(
            data=torch.zeros(
                self.qkv_size + self.indexer_size,
                self.input_size_per_partition,
                dtype=self.params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self.weight_loader_v2,
        )
        self.register_parameter("weight", weight)
        self.update_param_tp_status()
        self._loaded_shards: set[str] = set()

    @property
    def local_index_q_size(self) -> int:
        return self.local_index_q_heads * self.proxy_dim

    @property
    def local_index_k_size(self) -> int:
        return self.local_index_kv_heads * self.proxy_dim

    @property
    def local_index_z_size(self) -> int:
        return self.local_index_kv_heads * self.proxy_dim

    def validate_shard_id(self, loaded_shard_id: Any) -> TypeIs[str | None]:
        if loaded_shard_id in self.INDEXER_SHARD_IDS:
            return True
        return super().validate_shard_id(loaded_shard_id)

    def _indexer_source_offset(self, shard_id: str) -> tuple[int, int]:
        if shard_id == "index_q":
            return (
                self.index_group_start
                * (self.local_index_q_size // self.local_index_groups),
                self.local_index_q_size,
            )
        if shard_id in ("index_k", "index_z"):
            shard_index = self.tp_rank // self.index_kv_head_replicas
            return shard_index * self.local_index_k_size, self.local_index_k_size
        if shard_id == "index_g":
            return self.tp_rank * self.local_gate_size, self.local_gate_size
        raise ValueError(f"Unsupported sparse indexer shard: {shard_id}.")

    def _indexer_dest_offset(self, shard_id: str) -> int:
        offset = self.qkv_size
        if shard_id == "index_q":
            return offset
        offset += self.local_index_q_size
        if shard_id == "index_k":
            return offset
        offset += self.local_index_k_size
        if shard_id == "index_z":
            return offset
        offset += self.local_index_z_size
        if shard_id == "index_g":
            return offset
        raise ValueError(f"Unsupported sparse indexer shard: {shard_id}.")

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        if loaded_shard_id is None:
            raise NotImplementedError(
                "Step5 expects separate q/k/v/sparse_indexer_*/g_proj checkpoint "
                "tensors; a pre-fused qkv_indexer_proj weight is not supported."
            )
        if loaded_shard_id in self.INDEXER_SHARD_IDS:
            src_offset, shard_size = self._indexer_source_offset(loaded_shard_id)
            dst_offset = self._indexer_dest_offset(loaded_shard_id)
            shard = loaded_weight.narrow(0, src_offset, shard_size)
            param.data.narrow(0, dst_offset, shard_size).copy_(shard)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        self._loaded_shards.add(loaded_shard_id)

    def required_shard_ids(self) -> set[str]:
        return {"q", "k", "v", "index_q", "index_k", "index_z", "index_g"}

    def assert_fully_loaded(self, prefix: str) -> None:
        """Fail unless this load covered every shard, then forget the record.

        Tracking per load (instead of latching a flag on first success) is what
        makes weight reloading safe: a reload materializes fresh uninitialized
        storage for the whole parameter, so a shard the new weight stream omits
        would otherwise silently become garbage.
        """
        required = self.required_shard_ids()
        missing = sorted(required - self._loaded_shards)
        if missing:
            raise ValueError(
                f"Incomplete fused qkv+indexer weight for {prefix}: "
                f"missing shards {missing}."
            )
        self._loaded_shards.clear()

    def reset_load_state(self) -> None:
        self._loaded_shards.clear()

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.nn.functional.linear(hidden_states, self.weight)
        return combined[..., : self.qkv_size], combined[..., self.qkv_size :]

    def split_indexer(
        self,
        qkzg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index_q, index_k, index_z, gate = qkzg.split(
            [
                self.local_index_q_size,
                self.local_index_k_size,
                self.local_index_z_size,
                self.local_gate_size,
            ],
            dim=-1,
        )
        return index_q, index_k, index_z, gate


def _reset_fused_qkv_indexer_load_state(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, Step5FusedQKVIndexerLinear):
            submodule.reset_load_state()


def _validate_fused_qkv_indexer_weights(module: nn.Module) -> set[str]:
    loaded_params: set[str] = set()
    for name, submodule in module.named_modules():
        if not isinstance(submodule, Step5FusedQKVIndexerLinear):
            continue
        param_name = f"{name}.weight" if name else "weight"
        submodule.assert_fully_loaded(param_name)
        loaded_params.add(param_name)
    return loaded_params


class Step5SparseIndexerIndexTPLinear(ColumnParallelLinear):
    """Sparse-indexer linear sharded by its own tensor-parallel size."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype,
        index_tp_size: int,
        prefix: str = "",
    ) -> None:
        index_group_start, local_groups, _ = _step5_group_partition(
            total_groups=index_tp_size,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
        )
        if output_size <= 0 or output_size % index_tp_size != 0:
            raise ValueError(
                "Step5 sparse indexer output size must be divisible by "
                f"index_tp_size, got output_size={output_size}, "
                f"index_tp_size={index_tp_size}."
            )
        super().__init__(
            input_size,
            output_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=prefix,
            tp_rank=index_group_start // local_groups,
            tp_size=index_tp_size // local_groups,
        )
        self.index_group_start = index_group_start
        self.output_size_per_group = output_size // index_tp_size
        self.weight.data.zero_()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        start = self.index_group_start * self.output_size_per_group
        shard = loaded_weight.narrow(0, start, self.output_size_per_partition)
        param.data.copy_(shard)

    weight_loader_v2 = weight_loader

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return torch.nn.functional.linear(input_, self.weight, None), None


def _step5_use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel = vllm_config.parallel_config
    return (
        parallel.pipeline_parallel_size == 1
        and parallel.enable_expert_parallel
        and parallel.tensor_parallel_size > 1
        and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
    )


def _step5_sp_padding(hidden_states: torch.Tensor) -> torch.Tensor:
    padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        padding = get_forward_context().is_padding
        if padding is not None:
            padding = padding[: hidden_states.shape[0]]
    return sp_padding_mask(padding, hidden_states)


class Step5MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        sequence_parallel: bool = False,
        can_shard_sequence_parallel: bool = False,
        reduce_in_fp32: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.shard_sequence_parallel = (
            sequence_parallel
            and can_shard_sequence_parallel
            and envs.VLLM_STEP5_SHARD_SP_SHARED_EXPERT
            and tp_size > 1
            and hidden_size % tp_size == 0
            and intermediate_size % tp_size == 0
        )
        replicate = sequence_parallel and not self.shard_sequence_parallel
        self.reduce_in_fp32 = reduce_in_fp32
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=replicate,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False if self.shard_sequence_parallel else reduce_results,
            disable_tp=replicate,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.limit = None
        layer_idx = extract_layer_index(prefix)
        swiglu_limit = _per_layer_value(
            getattr(config, "swiglu_limits_shared", None),
            layer_idx,
            name="swiglu_limits_shared",
            default=None,
        )
        if swiglu_limit not in (None, 0):
            self.limit = swiglu_limit
            self.act_fn = SwigluStepAndMul(limit=self.limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shard_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        if self.shard_sequence_parallel:
            if self.reduce_in_fp32:
                output = output.float()
            output = sp_reduce_scatter(output)
        return output


def _step5_moe_reduce_policy(tp_size: int, dp_size: int) -> tuple[bool, bool]:
    """Return combined-reduce and per-path-reduce settings for Step5 MoE."""
    fuse_all_reduce = tp_size > 1 and dp_size == 1
    return fuse_all_reduce, not fuse_all_reduce


class Step5Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list[Any] | None = None,
        use_rope_layers: list[Any] | None = None,
        yarn_only_types: list[Any] | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step5SparseConfig | None = None,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if qkv_bias:
            raise NotImplementedError("Step5 attention_bias is not supported.")
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        sparse_indexer_base_rope_parameters = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        default_layer_type = (
            "sliding_attention" if self.layer_idx % 2 == 0 else "full_attention"
        )
        layer_type = _per_layer_value(
            layer_types,
            self.layer_idx,
            name="layer_types",
            default=default_layer_type,
        )
        enable_sliding_window = layer_type == "sliding_attention"
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            if not rope_theta:
                raise ValueError("Step5 rope_theta cannot be an empty list.")
            rope_theta = _per_layer_value(
                rope_theta,
                self.layer_idx,
                name="rope_theta",
                default=None,
            )

        self.rank = get_tensor_model_parallel_rank()
        if self.total_num_heads <= 0 or self.total_num_heads % tp_size != 0:
            raise ValueError(
                "Step5 attention heads must be positive and divisible by tensor "
                f"parallel size, got num_heads={self.total_num_heads}, "
                f"tp_size={tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads <= 0:
            raise ValueError(
                "Step5 attention requires a positive number of KV heads, got "
                f"{self.total_num_kv_heads}."
            )
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Step5 KV heads must be divisible by tensor parallel size "
                    "when sharded, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        else:
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    "Step5 KV heads must divide tensor parallel size when "
                    "replicated, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is None and hidden_size % self.total_num_heads != 0:
            raise ValueError(
                "Step5 hidden_size must be divisible by num_heads when head_dim "
                f"is omitted, got hidden_size={hidden_size}, "
                f"num_heads={self.total_num_heads}."
            )
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        if self.head_dim <= 0:
            raise ValueError(f"Step5 head_dim must be positive, got {self.head_dim}.")
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if (
            self.partial_rotary_factor <= 0.0
            or self.partial_rotary_factor > 1.0
            or self.rotary_dim <= 0
            or self.rotary_dim % 2 != 0
        ):
            raise ValueError(
                "Step5 partial_rotary_factor must produce a positive, even "
                "rotary dimension no larger than head_dim, got "
                f"head_dim={self.head_dim}, "
                f"partial_rotary_factor={self.partial_rotary_factor}, "
                f"rotary_dim={self.rotary_dim}."
            )
        if max_position is None or int(max_position) <= 0:
            raise ValueError(
                f"Step5 max_position must be a positive integer, got {max_position}."
            )
        max_position = int(max_position)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Pack Q/K/V and the per-head gate into one column-parallel GEMM when
        # KV heads are partitioned (not replicated) across TP ranks. The local
        # output layout is [Q, K, V, gate].
        self.fuse_qkv_gate = (
            use_head_wise_attn_gate
            and self.total_num_kv_heads >= tp_size
            and (
                quant_config is None
                or all(
                    isinstance(
                        LinearBase(
                            hidden_size,
                            width,
                            quant_config=quant_config,
                            prefix=f"{prefix}.{name}",
                            tp_rank=self.rank,
                            tp_size=tp_size,
                        ).quant_method,
                        UnquantizedLinearMethod,
                    )
                    for name, width in (
                        (
                            "qkv_proj",
                            (self.total_num_heads + 2 * num_kv_heads) * self.head_dim,
                        ),
                        ("g_proj", self.total_num_heads),
                    )
                )
            )
        )
        self.use_dsa_backend = sparse_config is not None and sliding_window is None
        attn_backend: type[AttentionBackend] | None
        self.sparse_summary_cache: Any = None
        if self.use_dsa_backend:
            from .sparse_attention import Step5DSAAttentionBackend

            if vllm_config is None:
                raise ValueError("Step5 DSA requires a complete VllmConfig.")
            _verify_step5_dsa_kv_transfer_compatibility(vllm_config)
            capability = current_platform.get_device_capability()
            if capability is None or not (
                Step5DSAAttentionBackend.supports_compute_capability(capability)
            ):
                raise NotImplementedError(
                    f"Step5 DSA requires SM90, SM100 or SM103; got {capability}."
                )
            if self.head_dim not in (128, 192):
                raise ValueError(
                    "Step5 DSA supports attention head dimensions 128 or 192, "
                    f"got {self.head_dim}."
                )
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "Step5 DSA does not support dual-batch overlap or "
                    "microbatching yet. Disable --enable-dbo and set "
                    "--ubatch-size 1."
                )
            sparse_config = typing.cast(Step5SparseConfig, sparse_config)
            _validate_step5_dsa_parallel_geometry(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
                index_tp_size=int(sparse_config.index_tp_size),
                tp_size=tp_size,
            )
            if not use_head_wise_attn_gate:
                raise NotImplementedError(
                    "Step5 DSA layers require use_head_wise_attn_gate=true."
                )
            proxy_dim = int(sparse_config.proxy_dim)
            self.qkv_indexer_proj = Step5FusedQKVIndexerLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                indexer_q_output_size=(
                    int(sparse_config.sparse_indexer_num_heads) * proxy_dim
                ),
                indexer_kv_output_size=(
                    int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim
                ),
                gate_output_size=self.total_num_heads,
                proxy_dim=proxy_dim,
                index_tp_size=int(sparse_config.index_tp_size),
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_indexer_proj",
            )
            self.params_dtype = self.qkv_indexer_proj.params_dtype
        elif self.fuse_qkv_gate:
            self.qkvg_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_heads,
                ],
                bias=False,
                quant_config=quant_config,
                # Keep the original prefix so quantization configuration that
                # targets qkv_proj continues to apply to the fused module.
                prefix=f"{prefix}.qkv_proj",
            )
            self.params_dtype = self.qkvg_proj.params_dtype
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.params_dtype = self.qkv_proj.params_dtype
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if rope_theta is not None:
            rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        sparse_indexer_base_rope_parameters.setdefault("rope_type", "default")
        if rope_theta is not None:
            sparse_indexer_base_rope_parameters["rope_theta"] = rope_theta
        sparse_indexer_base_rope_parameters["partial_rotary_factor"] = (
            partial_rotary_factor
        )

        self._sparse_indexer_base_rope_parameters = dict(
            sparse_indexer_base_rope_parameters
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )

        self.zero_centered = zero_centered
        self.q_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = make_step5_rms_norm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if (
            use_head_wise_attn_gate
            and not self.use_dsa_backend
            and not self.fuse_qkv_gate
        ):
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers,
                self.layer_idx,
                name="use_rope_layers",
                default=True,
            )
        )

        # Non-DSA layers pin FlashAttention for the fused SWA decode. The DSA
        # backend fixes the KV layout that every layer shares.
        if self.use_dsa_backend:
            attn_backend = Step5DSAAttentionBackend
        else:
            from .sparse_attention import Step5SplitKVFlashAttentionBackend

            attn_backend = Step5SplitKVFlashAttentionBackend
        sparse_impl_args: dict[str, Any] = {}
        if self.use_dsa_backend:
            dsa_vllm_config = typing.cast(VllmConfig, vllm_config)
            model_config = dsa_vllm_config.model_config
            scheduler_config = dsa_vllm_config.scheduler_config
            speculative_config = dsa_vllm_config.speculative_config
            sparse_impl_args = {
                "sparse_config": sparse_config,
                "max_model_len": int(model_config.max_model_len),
                "max_num_seqs": int(scheduler_config.max_num_seqs),
                "num_speculative_tokens": (
                    int(speculative_config.num_speculative_tokens)
                    if speculative_config is not None
                    and getattr(speculative_config, "method", None) == "mtp"
                    else 0
                ),
            }
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            # Step5 checkpoints do not quantize the KV cache.
            quant_config=None,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            attn_backend=attn_backend,
            **sparse_impl_args,
        )
        if self.use_dsa_backend:
            from .csa_tail_ring import Step5CSATailRing
            from .sparse_summary_cache import Step5SparseSummaryCacheLayer

            sparse_config = typing.cast(Step5SparseConfig, sparse_config)
            proxy_dim = int(sparse_config.proxy_dim)
            indexer_num_heads = int(sparse_config.sparse_indexer_num_heads)
            self.sparse_indexer_w = Step5SparseIndexerIndexTPLinear(
                self.hidden_size,
                indexer_num_heads,
                params_dtype=self.params_dtype,
                index_tp_size=int(sparse_config.index_tp_size),
                prefix=f"{prefix}.sparse_indexer_w",
            )
            # The sparse indexer uses weighted ReLU instead of scalable softmax,
            # so this checkpoint tensor is intentionally unused. Keep it
            # registered so weight loading accounts for every model parameter.
            self.ssmax_s = Parameter(
                torch.zeros(self.total_num_heads, dtype=torch.float32),
                requires_grad=False,
            )
            self.sparse_indexer_q_norm = make_step5_rms_norm(
                proxy_dim,
                eps=rms_norm_eps,
                zero_centered=self.zero_centered,
            )
            self.sparse_indexer_k_norm = Step5IndexerLayerNorm(
                proxy_dim,
                eps=rms_norm_eps,
            )
            static_forward_context = typing.cast(
                VllmConfig, vllm_config
            ).compilation_config.static_forward_context
            self.sparse_summary_cache = Step5SparseSummaryCacheLayer(
                prefix=f"{prefix}.attn.sparse_summary_cache",
                target_impl=self.attn.impl,
                static_forward_context=static_forward_context,
            )
            self.csa_tail_ring = Step5CSATailRing(
                prefix=f"{prefix}.attn.csa_tail_ring",
                main_layer_name=f"{prefix}.attn",
                num_speculative_tokens=sparse_impl_args["num_speculative_tokens"],
                static_forward_context=static_forward_context,
            )
            typing.cast(Any, self.attn.impl).csa_ring = self.csa_tail_ring
        self.max_position_embeddings = max_position
        self.sparse_attn = (
            Step5SparseAttention(self, typing.cast(Step5SparseConfig, sparse_config))
            if self.use_dsa_backend
            else None
        )

        self.fuse_swa_gate = (
            not self.use_dsa_backend
            and self.use_head_wise_attn_gate
            and self.params_dtype == torch.bfloat16
            and current_platform.is_device_capability(103)
            and typing.cast(Any, self.attn.impl).supports_fused_swa_decode()
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_dsa_backend:
            qkv, qkzg = self.qkv_indexer_proj(hidden_states)
            extra_dims = None
        elif self.fuse_qkv_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            qkv, extra_dims = qkvg.split(
                [self.q_size + 2 * self.kv_size, self.num_heads], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            qkzg = None
            extra_dims = None

        if self.fuse_swa_gate and extra_dims is None:
            extra_dims, _ = self.g_proj(hidden_states)

        attn_metadata, _, kv_cache, slot_mapping = get_attention_context(
            self.attn.layer_name
        )
        decode_slots = None
        speculative = (
            typing.cast(Any, self.attn.impl).num_speculative_tokens
            if self.sparse_attn is not None
            else 0
        )
        runtime_mode = get_forward_context().cudagraph_runtime_mode
        if (
            self.sparse_attn is not None
            and not speculative
            and _can_fuse_decode_aux(attn_metadata, runtime_mode)
        ):
            decode_slots = attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens]
        # MTP verify fuses the preparation only; its CSA update runs in the
        # attention, one program per request.
        fuse_prepare = decode_slots is not None or (
            speculative > 0
            and _is_uniform_verify(attn_metadata, runtime_mode, speculative)
        )
        kv_cache_dummy_dep = None
        prepared_indexer = None
        if slot_mapping is not None and kv_cache.numel() > 0:
            key_cache, value_cache = typing.cast(Any, self.attn.impl)._split_kv_cache(
                kv_cache
            )
            if (
                self.sparse_attn is not None
                and self.sparse_attn.fuse_qkv_preparation
                and fuse_prepare
                and 0 < qkv.shape[0] <= 128
            ):
                (q, k, v), prepared_indexer = self.sparse_attn.prepare_qkv(
                    positions,
                    hidden_states,
                    qkv,
                    qkzg,
                    slot_mapping,
                    key_cache,
                    value_cache,
                    decode_slots,
                )
            else:
                q, k, v = ops.fused_qknorm_rope_cache(
                    qkv,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.rotary_emb.cos_sin_cache,
                    positions,
                    slot_mapping,
                    key_cache,
                    value_cache,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.rotary_dim if self.use_rope else 0,
                    self.q_norm.variance_epsilon,
                    float(self.zero_centered),
                    self.attn.kv_cache_dtype,
                    self.attn._k_scale,
                    self.attn._v_scale,
                )
            kv_cache_dummy_dep = q.new_empty(0)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head.contiguous())
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head.contiguous())
            k = k_by_head.view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)
        if self.sparse_attn is not None:
            attn_output, extra_dims = self.sparse_attn.forward(
                positions,
                hidden_states,
                q,
                k,
                v,
                qkzg=qkzg,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                decode_slots=decode_slots,
                prepared_indexer=prepared_indexer,
            )
        elif self.fuse_swa_gate:
            assert extra_dims is not None
            key = k.view(-1, self.num_kv_heads, self.head_dim)
            value = v.view(-1, self.num_kv_heads, self.head_dim)
            if (
                kv_cache_dummy_dep is None
                and not self.attn.attn_backend.forward_includes_kv_cache_update
                and self.attn.kv_sharing_target_layer_name is None
            ):
                unified_kv_cache_update(key, value, self.attn.layer_name)
            attn_output = torch.empty(
                (q.shape[0], self.num_heads, self.head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            step5_swa_attention_with_output(
                q.view(-1, self.num_heads, self.head_dim),
                key,
                value,
                attn_output,
                self.attn.layer_name,
                extra_dims,
            )
            attn_output = attn_output.view(-1, self.q_size)
        elif kv_cache_dummy_dep is not None:
            # The fused prepare already wrote K/V; None keys skip the cache update.
            attn_output = self.attn(q, None, None)
        else:
            attn_output = self.attn(q, k, v)
        if extra_dims is None and self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)

        # SWA (fuse_swa_gate) and DSA attention apply the gate themselves.
        if (
            self.use_head_wise_attn_gate
            and not self.fuse_swa_gate
            and self.sparse_attn is None
        ):
            assert extra_dims is not None
            attn_output = headwise_sigmoid_gate(
                attn_output, extra_dims, slot_mapping=decode_slots
            )
        output, _ = self.o_proj(attn_output)
        return output


class Step5SparseAttention:
    def __init__(
        self,
        owner: Step5Attention,
        sparse_config: Step5SparseConfig,
    ) -> None:
        self.owner = owner
        self.sparse_config = sparse_config
        self.sparse_rotary_emb = self._build_sparse_rotary_emb()
        self._indexer_native_rope = not self.sparse_rotary_emb.enabled()
        owner.add_module("sparse_indexer_rotary_emb", self.sparse_rotary_emb)
        if not (
            owner.params_dtype == torch.bfloat16
            and int(sparse_config.proxy_dim) == 256
            and int(sparse_config.sparse_indexer_num_k_heads) == 1
            and self.sparse_rotary_emb.is_neox_style
            and self.sparse_rotary_emb.rotary_dim > 0
            and not self.sparse_rotary_emb.use_flashinfer
        ):
            raise NotImplementedError(
                "Step5 DSA requires BF16 params, proxy_dim=256, one sparse "
                "indexer K head and a non-FlashInfer NeoX indexer RoPE."
            )
        self.fuse_qkv_preparation = owner.head_dim == 192

    def prepare_qkv(
        self,
        positions,
        hidden_states,
        qkv,
        qkzg,
        slots,
        key_cache,
        value_cache,
        decode_slots,
    ):
        from vllm.models.step5.common.ops.qkv_indexer_norm_rope import (
            qkv_indexer_norm_rope,
        )

        owner = self.owner
        index_q, index_k, index_z, gate = owner.qkv_indexer_proj.split_indexer(qkzg)
        weights, _ = owner.sparse_indexer_w(hidden_states)
        impl = typing.cast(Any, owner.attn.impl)
        cache = impl.summary_cache if decode_slots is not None else None
        index_q, index_k, weights, q, k, v = qkv_indexer_norm_rope(
            index_q,
            index_k,
            weights,
            owner.sparse_indexer_q_norm.weight,
            owner.sparse_indexer_k_norm.weight,
            owner.sparse_indexer_k_norm.bias,
            self.sparse_rotary_emb.cos_sin_cache,
            positions.reshape(-1),
            owner.sparse_indexer_q_norm.variance_epsilon,
            owner.zero_centered,
            native_rope=self._indexer_native_rope,
            main_inputs=(
                qkv,
                owner.q_norm.weight,
                owner.k_norm.weight,
                owner.rotary_emb.cos_sin_cache,
                slots,
                key_cache,
                value_cache,
                owner.num_heads,
                owner.num_kv_heads,
                owner.rotary_dim if owner.use_rope else 0,
                owner.q_norm.variance_epsilon,
                float(owner.zero_centered),
                owner.attn._k_scale,
                owner.attn._v_scale,
            ),
            decode_cache=cache,
            decode_ring=impl.csa_ring if cache is not None else None,
            decode_z=index_z if cache is not None else None,
            decode_slots=decode_slots,
        )
        return (q, k, v), (
            index_q,
            index_k,
            weights,
            index_z.view(-1, 1, 256),
            gate,
            cache is not None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        qkzg: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        decode_slots: torch.Tensor | None = None,
        prepared_indexer=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if prepared_indexer is None:
            prepared_indexer = self._project_sparse_indexer(
                positions, hidden_states, qkzg, decode_slots=decode_slots
            )
        (
            dsa_proxy_query,
            dsa_proxy_key,
            dsa_proxy_weights,
            dsa_proxy_z,
            extra_dims,
            summary_updated,
        ) = prepared_indexer
        attn_output = self._forward_dsa_attention(
            query,
            key,
            value,
            dsa_proxy_query=dsa_proxy_query,
            dsa_proxy_key=dsa_proxy_key,
            dsa_proxy_weights=dsa_proxy_weights,
            dsa_proxy_z=dsa_proxy_z,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
            summary_updated=summary_updated,
            defer_inactive_clear=(
                decode_slots is not None and self.owner.use_head_wise_attn_gate
            ),
            # The attention applies the headwise gate while writing its output.
            gate=extra_dims if self.owner.use_head_wise_attn_gate else None,
        )
        return attn_output, extra_dims

    def _forward_dsa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dsa_proxy_query: torch.Tensor,
        dsa_proxy_key: torch.Tensor,
        dsa_proxy_weights: torch.Tensor,
        dsa_proxy_z: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor | None = None,
        summary_updated: bool = False,
        defer_inactive_clear: bool = False,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn = self.owner.attn
        hidden_size = attn.num_heads * attn.head_size_v
        output = torch.empty(
            (query.shape[0], hidden_size),
            dtype=query.dtype,
            device=query.device,
        )

        query = query.view(-1, attn.num_heads, attn.head_size)
        output = output.view(-1, attn.num_heads, attn.head_size_v)
        key = key.view(-1, attn.num_kv_heads, attn.head_size)
        value = value.view(-1, attn.num_kv_heads, attn.head_size_v)

        if (
            kv_cache_dummy_dep is None
            and not attn.attn_backend.forward_includes_kv_cache_update
            and attn.kv_sharing_target_layer_name is None
        ):
            unified_kv_cache_update(key, value, attn.layer_name)
        step5_dsa_attention_with_output(
            query,
            key,
            value,
            output,
            attn.layer_name,
            dsa_proxy_query,
            dsa_proxy_key,
            dsa_proxy_weights,
            dsa_proxy_z,
            summary_updated,
            defer_inactive_clear,
            gate,
        )
        return output.view(-1, hidden_size)

    def _sparse_indexer_rope_parameters(
        self,
        *,
        proxy_dim: int,
        rope_dim: int,
    ) -> dict[str, Any]:
        rope_parameters = dict(self.owner._sparse_indexer_base_rope_parameters)
        rope_type = str(rope_parameters.get("rope_type", "default")).strip().lower()
        if rope_type == "none":
            rope_parameters["rope_type"] = "default"
        rope_parameters["partial_rotary_factor"] = float(rope_dim) / float(proxy_dim)
        return rope_parameters

    def _build_sparse_rotary_emb(self) -> nn.Module:
        rope_dim = int(self.sparse_config.sparse_indexer_rope_dim)
        proxy_dim = int(self.sparse_config.proxy_dim)
        return get_rope(
            head_size=proxy_dim,
            max_position=self.owner.max_position_embeddings,
            rope_parameters=self._sparse_indexer_rope_parameters(
                proxy_dim=proxy_dim,
                rope_dim=rope_dim,
            ),
            dtype=self.owner.params_dtype,
        )

    def _project_sparse_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        qkzg: torch.Tensor,
        *,
        decode_slots: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        from vllm.models.step5.common.ops.indexer_norm_rope import indexer_norm_rope

        owner = self.owner
        index_q, index_k, index_z, extra_dims = owner.qkv_indexer_proj.split_indexer(
            qkzg
        )
        weights, _ = owner.sparse_indexer_w(hidden_states)
        impl = typing.cast(Any, owner.attn.impl)
        cache = impl.summary_cache if decode_slots is not None else None
        index_q, index_k, weights = indexer_norm_rope(
            index_q,
            index_k,
            weights,
            owner.sparse_indexer_q_norm.weight,
            owner.sparse_indexer_k_norm.weight,
            owner.sparse_indexer_k_norm.bias,
            self.sparse_rotary_emb.cos_sin_cache,
            positions.reshape(-1),
            owner.sparse_indexer_q_norm.variance_epsilon,
            owner.zero_centered,
            native_rope=self._indexer_native_rope,
            num_groups=int(owner.num_kv_heads),
            decode_cache=cache,
            decode_ring=impl.csa_ring if cache is not None else None,
            decode_z=index_z if cache is not None else None,
            decode_slots=decode_slots if cache is not None else None,
        )
        index_z = index_z.view(int(hidden_states.shape[0]), 1, 256)
        return index_q, index_k, weights, index_z, extra_dims, cache is not None


class Step5MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.gate = GateLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        if not config.use_moe_router_bias:
            raise ValueError("Step5 MoE currently requires use_moe_router_bias=true.")
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        if not config.need_fp32_gate:
            raise ValueError(
                "Step5 MoE requires need_fp32_gate=true for stable router logits."
            )

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            if not isinstance(swiglu_limit, (int, float)):
                raise ValueError(
                    "Step5 swiglu_limits entries must be numeric, got "
                    f"{swiglu_limit!r}."
                )
            swiglu_limit = float(swiglu_limit)
            if swiglu_limit != 7.0:
                raise ValueError(
                    "Step5 fused MoE supports only swiglu_limit=7.0, got "
                    f"{swiglu_limit}."
                )
            activation = "swiglustep"

        self.fuse_all_reduce, reduce_results = _step5_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )

        kernel_config = getattr(vllm_config, "kernel_config", None)
        self.use_mega_moe = (
            getattr(kernel_config, "moe_backend", "auto") == "deep_gemm_mega_moe"
        )
        self.use_sequence_parallel = _step5_use_sequence_parallel(vllm_config)
        if self.use_mega_moe:
            self.fuse_all_reduce, reduce_results = False, False

        self.share_expert = Step5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert",
            sequence_parallel=self.use_sequence_parallel,
            can_shard_sequence_parallel=self.use_mega_moe,
            reduce_in_fp32=True,
        )
        # Step5 selects experts with a correction bias but computes routed
        # weights from the unbiased sigmoid scores.
        expert_kwargs = dict(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
        )
        if self.use_mega_moe:
            from vllm.models.step5.nvidia.mega_moe import Step5MegaMoE

            self.experts = Step5MegaMoE(
                vllm_config,
                **expert_kwargs,
                activation_clamp=swiglu_limit or None,
                sequence_parallel=self.use_sequence_parallel,
            )
        else:
            self.experts = FusedMoEFactory(
                **expert_kwargs,
                reduce_results=reduce_results,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                router_logits_dtype=torch.float32,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = self.share_expert(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        expert_kwargs = {"padding": padding} if padding is not None else {}
        routed_output = self.experts(
            hidden_states=hidden_states, router_logits=router_logits, **expert_kwargs
        )

        return shared_output, routed_output


class Step5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # Preserve the configured SWA window before clearing the per-layer
        # cache copy; layer_types determines which layers use it.
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            # Multimodal checkpoints keep sliding_window in the text config.
            sliding_window = getattr(
                vllm_config.model_config.hf_text_config, "sliding_window", None
            )
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy.copy(cache_config)
            cache_config.sliding_window = None
        sparse_config = get_step5_sparse_config(config)
        if not checkpoint_has_step5_sparse_config(config):
            raise NotImplementedError(
                "Step5 requires a checkpoint with a sparse_config section; "
                "dense-only checkpoints are not supported."
            )
        use_dsa_for_layer = (
            sparse_config is not None
            and _is_step5_full_attention_layer(config, layer_idx, sparse_config)
        )
        if config.att_impl_type == "GQA":
            norm_dtype = get_step5_norm_dtype(config)
            num_attention_heads = None
            num_attention_groups = None
            head_dim = None
            layer_types = _get_step5_layer_types(config)
            if (
                getattr(config, "attention_other_setting", None)
                and layer_idx < len(layer_types)
                and layer_types[layer_idx]
                == config.attention_other_setting["attention_type"]
            ):
                num_attention_heads = config.attention_other_setting[
                    "num_attention_heads"
                ]
                num_attention_groups = config.attention_other_setting[
                    "num_attention_groups"
                ]
                head_dim = config.attention_other_setting["head_dim"]
            partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
            partial_rotary_factor = float(
                _per_layer_value(
                    partial_rotary_factors,
                    layer_idx,
                    name="partial_rotary_factors",
                    default=1.0,
                )
            )
            max_position = getattr(config, "max_position_embeddings", None)
            if max_position is None:
                max_position = vllm_config.model_config.max_model_len
            self.self_attn = Step5Attention(
                hidden_size=self.hidden_size,
                num_heads=num_attention_heads
                if num_attention_heads
                else config.num_attention_heads,
                max_position=max_position,
                num_kv_heads=num_attention_groups
                if num_attention_groups
                else config.num_attention_groups,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
                cache_config=cache_config,
                quant_config=quant_config,
                rope_scaling=getattr(config, "rope_scaling", None),
                sliding_window=sliding_window,
                use_head_wise_attn_gate=getattr(
                    config, "use_head_wise_attn_gate", False
                ),
                layer_types=layer_types,
                use_rope_layers=getattr(config, "use_rope_layers", []),
                yarn_only_types=getattr(config, "yarn_only_types", []),
                swa_num_attention_heads=getattr(
                    config, "swa_num_attention_heads", None
                ),
                partial_rotary_factor=partial_rotary_factor,
                prefix=f"{prefix}.self_attn",
                zero_centered=config.zero_centered,
                vllm_config=vllm_config,
                sparse_config=sparse_config if use_dsa_for_layer else None,
                norm_dtype=norm_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )
        self.use_sequence_parallel = _step5_use_sequence_parallel(vllm_config)
        if self.use_sequence_parallel:
            self.self_attn.o_proj.reduce_results = False
        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
            and not self.use_sequence_parallel
        )

        moe_layers_idx = _get_step5_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = Step5MoE(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
                sequence_parallel=self.use_sequence_parallel,
                can_shard_sequence_parallel=True,
            )
        self.input_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = make_step5_rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )

    def _add_for_residual(self, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            return add_fp32(in1, in2)
        return in1 + in2

    def _cast_for_param_op(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.bfloat16)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_moe:
            shared_output, moe_output = self.moe(hidden_states, padding=padding)
            combined = (
                moe_output
                if shared_output is None
                else self._add_for_residual(moe_output, shared_output)
            )
            # Reduce once when both branches defer reduction; otherwise their
            # outputs are already reduced.
            if self.moe.fuse_all_reduce:
                if self.use_fused_all_reduce:
                    combined = self.tp_group.all_reduce(combined)
                else:
                    combined = tensor_model_parallel_all_reduce(combined)
            return combined
        return self.mlp(hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp32_residual_connection and isinstance(
            self.input_layernorm, Step5RMSNorm
        ):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(
                    hidden_states, out_dtype=torch.bfloat16
                )
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual, out_dtype=torch.bfloat16
                )
        else:
            if residual is not None:
                hidden_states = self._add_for_residual(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self._cast_for_param_op(hidden_states)

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)
        if self.fp32_residual_connection and isinstance(
            self.post_attention_layernorm, Step5RMSNorm
        ):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual, out_dtype=torch.bfloat16
            )
        else:
            hidden_states = self._cast_for_residual(hidden_states)
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self._cast_for_param_op(hidden_states)

        ffn_output = self._forward_ffn(hidden_states, padding=padding)
        return ffn_output, residual


class Step5Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.use_sequence_parallel = _step5_use_sequence_parallel(vllm_config)
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.moe_num_experts = config.moe_num_experts

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step5DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_step5_norm_dtype(config)
            self.norm = make_step5_rms_norm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=config.zero_centered,
                dtype=norm_dtype,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self._cast_for_residual(hidden_states)
        padding = None
        if self.use_sequence_parallel:
            padding = _step5_sp_padding(hidden_states)
            hidden_states = sp_shard(hidden_states)
        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, residual, padding=padding
            )
        if residual is not None:
            hidden_states = layer._add_for_residual(hidden_states, residual)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        config = self.config
        if config.num_attention_groups <= 1:
            raise ValueError(
                "Step5 weight loading currently supports only GQA "
                "(num_attention_groups > 1)."
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Quantized expert wrappers insert base_layer into parameter names.
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        # Packed 3D format: .moe.gate_proj.weight [num_experts, out, in]. Each
        # entry is a substring of its scale names (weight_scale_inv, ...), so
        # it maps those too.
        expert_params_mapping = [
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.gate_proj.weight",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.up_proj.weight",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight",
                ".moe.down_proj.weight",
                "w2",
            ),
        ]

        expert_params_mapping += [
            (
                param.replace("_weight", "_input_scale"),
                weight.replace(".weight", ".input_scale"),
                shard,
            )
            for param, weight, shard in expert_params_mapping
        ]

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        def _load_weight(name: str, loaded_weight: torch.Tensor) -> set[str]:
            loaded_params: set[str] = set()
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            if full_name.startswith("model.layers."):
                layer_prefix = ".".join(full_name.split(".", 3)[:3])
                layer_idx = extract_layer_index(layer_prefix)
                if layer_idx >= config.num_hidden_layers:
                    return loaded_params

            remapped_name = maybe_remap_kv_scale_name(local_name, params_dict)
            if remapped_name is None:
                return loaded_params
            local_name = remapped_name

            if ".moe.experts." in local_name:
                if is_pp_missing_parameter(local_name, self):
                    return loaded_params
                if local_name not in params_dict:
                    raise NotImplementedError(
                        f"Unsupported Step5 MoE checkpoint tensor {name!r}: only "
                        "the packed 3D expert format is supported."
                    )
                param = params_dict[local_name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )
                loaded_params.add(local_name)
                return loaded_params
            for param_name, weight_name, stacked_shard_id in _STEP5_WEIGHT_SHARDS:
                if weight_name not in local_name:
                    continue
                if any(
                    disable_moe_stacked_param in local_name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                replaced_name = local_name.replace(weight_name, param_name)
                if is_pp_missing_parameter(replaced_name, self):
                    continue
                if replaced_name not in params_dict:
                    continue
                param = params_dict[replaced_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(replaced_name)
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    if weight_name not in local_name:
                        continue
                    replaced_name = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(replaced_name, self):
                        continue
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    param = params_dict[replaced_name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.moe_num_experts
                    if (
                        loaded_weight.ndim == 0
                        or loaded_weight.shape[0] != moe_expert_num
                    ):
                        raise ValueError(
                            "Step5 expert tensor has an unexpected leading "
                            f"dimension: expected {moe_expert_num}, got "
                            f"{tuple(loaded_weight.shape)} for {replaced_name}."
                        )
                    for expert_id in range(moe_expert_num):
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            replaced_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(replaced_name)
                    break
                else:
                    if is_pp_missing_parameter(local_name, self):
                        return loaded_params
                    if "expert_bias" in local_name:
                        logger.warning_once("Ignoring unsupported expert_bias weight.")
                        return loaded_params
                    if local_name not in params_dict:
                        return loaded_params
                    param = params_dict[local_name]
                    fallback_weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    try:
                        fallback_weight_loader(param, loaded_weight)
                    except Exception:
                        logger.error(
                            "Failed to load %s: checkpoint shape=%s, "
                            "parameter shape=%s",
                            local_name,
                            tuple(loaded_weight.shape),
                            tuple(param.shape),
                        )
                        raise

                    loaded_params.add(local_name)
            return loaded_params

        for name, loaded_weight in skip_spec_layers(weights, config):
            loaded_params.update(_load_weight(name, loaded_weight))
        return loaded_params


class Step5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsLoRA):
    # Required so quantization exclude lists match fused module prefixes.
    packed_modules_mapping = STEP5_PACKED_MODULES_MAPPING

    hf_to_vllm_mapper = (
        WeightsMapper(
            orig_to_new_regex={
                re.compile(r"^vit_large_projector\.weight$"): None,
            },
        )
        | STEP5_SHARE_EXPERT_MAPPER
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        if (
            vllm_config.lora_config is not None
            and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        ):
            raise ValueError(
                "Step5 LoRA is not supported with moe_backend='deep_gemm_mega_moe'"
            )
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = Step5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.num_moe_layers: int
        self.num_expert_groups: int
        self.num_shared_experts: int
        self.num_logical_experts: int
        self.num_physical_experts: int
        self.num_local_physical_experts: int
        self.num_routed_experts: int
        self.num_redundant_experts: int
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.moe_layers: list[Any] = []
        example_layer: Step5MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step5DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, Step5MoE):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)

        _set_step5_moe_protocol_metadata(self, example_layer)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        hidden_states = self.model.norm(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step5 EPLB cannot change the number of local physical experts: "
                f"expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, Step5DecoderLayer):
                continue
            moe = getattr(layer, "moe", None)
            if not isinstance(moe, Step5MoE):
                continue
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.mtp_validation import (
            is_mtp_completeness_check_enabled,
        )

        validate_completeness = is_mtp_completeness_check_enabled()
        if validate_completeness:
            _reset_fused_qkv_indexer_load_state(self.model)
        skip_prefixes = ["vision_model."]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_prefix={prefix: None for prefix in skip_prefixes}
        )
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=mapper)
        if validate_completeness:
            loaded_params.update(
                f"model.{name}"
                for name in _validate_fused_qkv_indexer_weights(self.model)
            )
        return loaded_params
