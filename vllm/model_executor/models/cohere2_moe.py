# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import CohereConfig

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from .commandr import LayerNorm
from .interfaces import EagleModelMixin, SupportsEagle3, SupportsPP, SupportsQuant
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _dual_stream_layer(layer_name: LayerNameType) -> "Cohere2MoeDecoderLayer":
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    assert isinstance(layer, Cohere2MoeDecoderLayer)
    return layer


def _cohere2_mlp_fork(
    hidden_states: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    return _dual_stream_layer(layer_name)._fork_mlp(hidden_states)


def _cohere2_mlp_fork_fake(
    hidden_states: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _cohere2_mlp_join(
    attn_output: torch.Tensor, mlp_output: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    return _dual_stream_layer(layer_name)._join_mlp(attn_output, mlp_output)


def _cohere2_mlp_join_fake(
    attn_output: torch.Tensor, mlp_output: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    return torch.empty_like(attn_output)


# The MLP/MoE branch is forked onto a side stream before attention and joined
# after o_proj. Fork and join are opaque to Dynamo (a traced stream switch
# would be dropped), while attention stays a regular splitting op, so piecewise
# CUDA graphs keep working for prefill and mixed batches.
direct_register_custom_op(
    op_name="cohere2_mlp_fork",
    op_func=_cohere2_mlp_fork,
    fake_impl=_cohere2_mlp_fork_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
direct_register_custom_op(
    op_name="cohere2_mlp_join",
    op_func=_cohere2_mlp_join,
    fake_impl=_cohere2_mlp_join_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def is_prefix_dense_layer(config: CohereConfig, layer_idx: int) -> bool:
    """True when layer_idx lies in the contiguous dense MLP prefix."""
    if layer_idx >= len(config.mlp_layer_types):
        return False
    return all(t == "dense" for t in config.mlp_layer_types[: layer_idx + 1])


@torch.compile(backend=current_platform.simple_compile_backend)
def token_choice_with_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Sigmoid -> top-k (-> renormalize) custom routing for Cohere2Moe."""
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.float().sigmoid()
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def make_norm(config: CohereConfig) -> nn.Module:
    """RMSNorm when config.rms_norm_eps is set, otherwise Cohere LayerNorm with
    config.layer_norm_eps."""
    rms_eps = getattr(config, "rms_norm_eps", None)
    if rms_eps is not None:
        return RMSNorm(config.hidden_size, eps=rms_eps)
    return LayerNorm(param_shape=(config.hidden_size,), eps=config.layer_norm_eps)


class Cohere2MoeMLP(nn.Module):
    """Cohere MLP used as shared experts in the MoE block."""

    def __init__(
        self,
        config: CohereConfig,
        intermediate_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Cohere2MoeAttention(nn.Module):
    """Cohere MoE attention with sliding-window interleave."""

    def __init__(
        self,
        config: CohereConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(
            config, "model_max_length", None
        ) or getattr(config, "max_position_embeddings", 8192)
        assert isinstance(self.max_position_embeddings, int)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        self.sliding_window = None
        layer_types = getattr(config, "layer_types", None)
        if (
            layer_types is not None
            and layer_types[self.layer_idx] == "sliding_attention"
        ):
            self.sliding_window = config.sliding_window + 1

        # Prefix-dense layers have full attention (no sliding window). When
        # prefix_dense_sliding_window_pattern == 1, they keep RoPE even though
        # they are not sliding-window layers.
        prefix_dense_sliding_window_pattern = getattr(
            config, "prefix_dense_sliding_window_pattern", 1
        )
        self.force_rope = bool(
            is_prefix_dense_layer(config, self.layer_idx)
            and prefix_dense_sliding_window_pattern == 1
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.sliding_window or self.force_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Cohere2Moe(nn.Module):
    """Tensor-parallel MoE block for Cohere2Moe with shared experts."""

    def __init__(
        self,
        config: CohereConfig,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        if (
            hasattr(config, "expert_selection_fn")
            and config.expert_selection_fn == "sigmoid"
        ):
            self.custom_routing_function = token_choice_with_bias
        else:
            self.custom_routing_function = None

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            params_dtype=params_dtype,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: Cohere2MoeMLP | None
        if hasattr(config, "num_shared_experts") and config.num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                config=config,
                intermediate_size=config.intermediate_size * config.num_shared_experts,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )
            self.shared_expert_combination_strategy = getattr(
                config, "shared_expert_combination_strategy", "sum"
            )
            assert self.shared_expert_combination_strategy in ("average", "sum"), (
                "shared_expert_combination_strategy must be one of ['average', 'sum']"
            )
        else:
            self.shared_experts = None
            self.shared_expert_combination_strategy = None

        # Routed-only MoE leaves its TP all-reduce to the decoder layer, which
        # reduces the attention and MoE outputs together.
        defer_all_reduce = self.tp_size > 1 and self.shared_experts is None
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=params_dtype,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=f"{prefix}.experts",
            custom_routing_function=self.custom_routing_function,
            shared_experts=self.shared_experts,
            reduce_results=not defer_all_reduce,
        )
        # The deferral is dropped when the MoE runner must reduce itself
        # (e.g. EP all2all or sequence parallel).
        self.output_is_reduced = not (
            defer_all_reduce and self.experts.moe_config.skip_final_all_reduce
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        # FusedMoEFactory handles shared expert overlap internally and returns
        # shared_output + routed_output when shared_experts is set.
        final_hidden_states = self.experts(hidden_states, router_logits)
        if self.shared_expert_combination_strategy == "average":
            final_hidden_states = final_hidden_states / 2
        return final_hidden_states.view(orig_shape)


def _make_dual_stream(vllm_config: VllmConfig) -> torch.cuda.Stream | None:
    """Side stream shared by all decoder layers, or None when the dual stream
    is off or unsupported."""
    if not envs.VLLM_COHERE2_ATTN_MOE_DUAL_STREAM:
        return None
    if not current_platform.is_cuda():
        logger.warning_once(
            "VLLM_COHERE2_ATTN_MOE_DUAL_STREAM is only supported on CUDA; ignoring it."
        )
        return None
    if vllm_config.parallel_config.use_ubatching:
        # Each layer holds a single fork/join state, so microbatches would race.
        logger.warning_once(
            "VLLM_COHERE2_ATTN_MOE_DUAL_STREAM is not supported with "
            "microbatching (DBO); ignoring it."
        )
        return None
    logger.info_once(
        "Cohere2MoE attention/MoE dual stream enabled (token threshold=%d).",
        envs.VLLM_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD,
    )
    # Dedicated stream: the global aux_stream() is used by the shared-expert
    # and LoRA overlaps inside the MoE branch.
    return torch.cuda.Stream()


@dataclasses.dataclass
class _DualStreamState:
    """Dual-stream state of one decoder layer.

    ``stream`` is the model's single side stream, shared by every layer; the
    events and ``join_pending`` are per layer, since a layer has at most one
    fork in flight. DBO would need one state per ubatch."""

    stream: torch.cuda.Stream
    token_threshold: int
    fork_event: torch.cuda.Event = dataclasses.field(default_factory=torch.cuda.Event)
    join_event: torch.cuda.Event = dataclasses.field(default_factory=torch.cuda.Event)
    join_pending: bool = False


class Cohere2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: CohereConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        moe_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)

        self.self_attn = Cohere2MoeAttention(
            config,
            cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if config.mlp_layer_types[self.layer_idx] == "dense":
            self.mlp = Cohere2MoeMLP(
                config=config,
                intermediate_size=getattr(
                    config, "prefix_dense_intermediate_size", config.intermediate_size
                ),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Cohere2Moe(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        self.mlp_output_is_reduced = (
            isinstance(self.mlp, Cohere2Moe) and self.mlp.output_is_reduced
        )

        self.input_layernorm = make_norm(config)
        self.uses_rms_norm = isinstance(self.input_layernorm, RMSNorm)

        if moe_stream is not None and not self._allows_dual_stream():
            logger.warning_once(
                "Cohere2MoE dual stream disabled for MoE layers that reduce "
                "internally under TP (shared experts, EP all2all or sequence "
                "parallel)."
            )
            moe_stream = None
        self.dual_stream = (
            _DualStreamState(
                stream=moe_stream,
                token_threshold=envs.VLLM_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD,
            )
            if moe_stream is not None
            else None
        )
        self.layer_name = prefix
        if self.dual_stream is not None:
            static_forward_context = (
                get_current_vllm_config().compilation_config.static_forward_context
            )
            if prefix in static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            static_forward_context[prefix] = self

    def _allows_dual_stream(self) -> bool:
        """With the dual stream the only TP collective in a layer is the
        all-reduce of attention + MLP after the join. A MoE that reduces
        internally would let the attention all-reduce run while the MoE stream
        is still communicating.

        EP all2all kernels stay single-stream at any TP size (including DP + EP
        at TP1): all2all on the side stream, and next to DBO, is not
        validated."""
        if (
            isinstance(self.mlp, Cohere2Moe)
            and self.mlp.experts.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            return False
        return (
            get_tensor_model_parallel_world_size() == 1
            or not self.mlp_output_is_reduced
        )

    def _fork_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self.dual_stream
        assert state is not None
        assert not state.join_pending, "Cohere2MoE dual-stream fork was not joined"
        # A fork must not span a piecewise CUDA graph boundary: the segment
        # captured before attention would end with an unjoined stream. Prefill
        # and mixed batches use piecewise graphs, so they run the MLP inline.
        if (
            hidden_states.shape[0] > state.token_threshold
            or get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            or BreakableCUDAGraphCapture.is_active()
        ):
            return self.mlp(hidden_states)

        logger.info_once("Cohere2MoE attention/MoE dual stream is active.")
        # Inductor may reuse the storage of hidden_states once this op returns
        # (bypassing the caching allocator), so the MoE stream reads a copy.
        mlp_input = hidden_states.clone()
        state.fork_event.record(torch.cuda.current_stream())
        with torch.cuda.stream(state.stream):
            state.fork_event.wait()
            mlp_output = self.mlp(mlp_input)
            state.join_event.record()
        mlp_input.record_stream(state.stream)
        # mlp_output needs no record_stream: its block can only be reused by a
        # side-stream allocation, and side-stream work always runs after a later
        # fork_event, recorded on the main stream after every use of mlp_output.
        # Keep it that way: no side-stream work outside a fork.
        state.join_pending = True
        return mlp_output

    def _join_mlp(
        self, attn_output: torch.Tensor, mlp_output: torch.Tensor
    ) -> torch.Tensor:
        state = self.dual_stream
        assert state is not None
        if state.join_pending:
            torch.cuda.current_stream().wait_event(state.join_event)
            state.join_pending = False
        return attn_output + mlp_output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_rms_norm:
            residual = hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.dual_stream is not None:
            layer_name = _encode_layer_name(self.layer_name)
            hidden_states_mlp = torch.ops.vllm.cohere2_mlp_fork(
                hidden_states, layer_name
            )
            hidden_states_attention = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            # Only layers that reduce attention + MLP together keep a
            # dual_stream (see __init__), so the join can return the sum.
            parallel_block_output = tensor_model_parallel_all_reduce(
                torch.ops.vllm.cohere2_mlp_join(
                    hidden_states_attention, hidden_states_mlp, layer_name
                )
            )
        else:
            hidden_states_attention = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            hidden_states_mlp = self.mlp(hidden_states)
            if self.mlp_output_is_reduced:
                parallel_block_output = (
                    tensor_model_parallel_all_reduce(hidden_states_attention)
                    + hidden_states_mlp
                )
            else:
                parallel_block_output = tensor_model_parallel_all_reduce(
                    hidden_states_attention + hidden_states_mlp
                )

        if self.uses_rms_norm:
            # The residual add is fused into the next RMSNorm (or the final norm).
            return parallel_block_output, residual
        return residual + parallel_block_output, residual


@support_torch_compile
class Cohere2MoeModel(nn.Module, EagleModelMixin):
    """Transformer decoder for Cohere2Moe."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self._moe_stream = _make_dual_stream(vllm_config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Cohere2MoeDecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
                moe_stream=self._moe_stream,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = make_norm(config)
        self.uses_rms_norm = isinstance(self.norm, RMSNorm)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            start_hidden_states = hidden_states
            if self.uses_rms_norm and residual is None:
                # The first layer carries hidden_states as the residual, which
                # later fused RMSNorm calls update in place.
                start_hidden_states = hidden_states.clone()
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
                self.start_layer,
                start_hidden_states,
                residual if self.uses_rms_norm else None,
            )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx + 1,
                hidden_states,
                residual if self.uses_rms_norm else None,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Cohere2MoeForCausalLM(nn.Module, SupportsPP, SupportsQuant, SupportsEagle3):
    is_text_generation_model = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            # .experts.gate_up_proj must be handled by MoERunner.load_weights for EP
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
        orig_to_new_prefix={"lm_head.": None},
    )
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.unpadded_vocab_size = config.vocab_size
        self.quant_config = quant_config
        self.logits_scale = config.logit_scale
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size, scale=self.logits_scale
        )
        self.model = Cohere2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.model.embed_tokens, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
