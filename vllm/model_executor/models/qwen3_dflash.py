# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import io
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_SLIDING_ATTENTION = "sliding_attention"


def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return override
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == _SLIDING_ATTENTION


def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
    """Whether the draft needs a non-causal-capable backend, resolved from config
    (config mirror of the model's ``get_draft_attn_causal``, usable pre-build)."""
    return not all(
        _dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)
    )


def _get_dflash_fc_input_size(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    config = spec_config.draft_model_config.hf_config
    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    num_features_to_use = len(aux_layers) if aux_layers else config.num_hidden_layers
    target_hidden_size = (
        getattr(config, "target_hidden_size", None) or config.hidden_size
    )
    return target_hidden_size * num_features_to_use


class ContextKVStrategy(str, enum.Enum):
    """DFlash context-KV projection strategy (quantized-drafter support).

    ``precompute_and_store_context_kv`` projects the target hidden states to
    K/V with every drafter layer. For unquantized drafters this is a single
    fused GEMM over the raw K/V rows. Quantized drafters cannot feed their
    packed weights to a bare ``F.linear`` (see #51581); we pick a strategy
    from the layers' ``quant_method`` metadata before ever touching ``.weight``
    and build the fused buffers lazily after ``process_weights_after_loading``.

    - FUSED:          all layers unquantized -> slice raw weights + one F.linear
    - SCALED_MM:      simple FP8 (non-Marlin) on a cutlass-fp8 platform -> keep
                      the quantized weights + per-column scales + one fused
                      W8A8 GEMM (fusion and quantization both preserved)
    - FUSED_DEQUANT:  simple FP8 but no quantized-GEMM primitive -> dequant to
                      the compute dtype + one fused F.linear (fusion preserved)
    - PER_LAYER:      grouped-int4/NVFP4/MXFP4/Marlin/unknown -> per-layer
                      ``quant_method.apply`` (correct for every scheme, keeps
                      the per-layer quantization)
    """

    FUSED = "fused"
    SCALED_MM = "scaled_mm"
    FUSED_DEQUANT = "dequant"
    PER_LAYER = "per_layer"


# Schemes whose scale can be folded into a per-output-column vector
# (per-tensor / per-channel). Grouped(int4)/block(NVFP4/MXFP4)/Marlin scales
# cannot be folded and must go through the per-layer path.
_SIMPLE_FUSABLE_QUANT_METHODS = (
    "Fp8LinearMethod",
    "Fp8PerTensorOnlineLinearMethod",
)


def _is_simple_fusable(proj: nn.Module) -> bool:
    """Simple FP8 per-tensor/per-channel (non-Marlin, non-block)."""
    method = getattr(proj, "quant_method", None)
    if method is None or isinstance(method, UnquantizedLinearMethod):
        return True
    if type(method).__name__ not in _SIMPLE_FUSABLE_QUANT_METHODS:
        return False
    return not getattr(method, "use_marlin", False) and not getattr(
        method, "block_quant", False
    )


def _decide_context_kv_strategy(
    projections: Iterable[nn.Module],
) -> ContextKVStrategy:
    """Pick the context-KV projection strategy.

    Never touches ``.weight``, so packed/GPTQ/AWQ layers are safe to inspect.
    """
    projections = list(projections)
    methods = [getattr(p, "quant_method", None) for p in projections]

    # Unquantized: original fast path (zero change).
    if all(m is None or isinstance(m, UnquantizedLinearMethod) for m in methods):
        return ContextKVStrategy.FUSED

    # Simple FP8: prefer a fused quantized GEMM; fall back to a fused dequant.
    if all(_is_simple_fusable(p) for p in projections):
        if cutlass_fp8_supported():
            return ContextKVStrategy.SCALED_MM
        return ContextKVStrategy.FUSED_DEQUANT

    # Grouped/block/Marlin/mixed/unknown: per-layer quantized GEMMs.
    return ContextKVStrategy.PER_LAYER


def _kv_scale_vector(proj: nn.Module, q_size: int, kv_size: int) -> torch.Tensor:
    """Per-output-row scale of the K/V rows of a quantized QKV projection
    (``[2 * kv_size]``).

    - per-tensor (1 or 3 scales): broadcast over the K/V rows
    - per-channel: slice the K/V rows
    """
    s = getattr(proj, "weight_scale", None)
    if s is None:
        return torch.ones(2 * kv_size, device=proj.weight.device)
    s = s.reshape(-1)
    if s.numel() == 1:
        return s[0].expand(2 * kv_size)
    if s.numel() == 3:  # per-shard per-tensor scales: [q, k, v]
        return torch.cat([s[1].expand(kv_size), s[2].expand(kv_size)])
    return s[q_size:]  # per-channel: K/V rows


def _project_kv_per_layer(
    normed: torch.Tensor,
    kv_projections: Iterable[tuple[nn.Module, int | None]],
    *,
    per_layer_input: bool = False,
    stack: bool = False,
) -> torch.Tensor:
    """Per-layer quantized K/V projection (correct for every quant scheme).

    ``kv_projections`` maps each layer to ``(projection, q_size)``; ``q_size``
    is the number of Q rows/columns to drop, or ``None`` for K-only projections
    that have no Q rows. ``per_layer_input=True`` feeds each layer its own slice
    of a layer-major (grouped) input (laguna); otherwise all layers share
    ``normed``. ``stack=True`` keeps the layer axis (laguna bmm path), otherwise
    layers are concatenated along the feature axis (qwen3 / gemma4 layouts).
    Each projection's bias is applied here (packed kernels return it separately).
    """
    outs = []
    for i, (proj, q_size) in enumerate(kv_projections):
        src = normed[i] if per_layer_input else normed
        out, bias = proj(src)
        if bias is not None:
            out = out + bias
        outs.append(out if q_size is None else out[..., q_size:])
    return torch.stack(outs, dim=0) if stack else torch.cat(outs, dim=-1)


def _resolve_layer_attention(
    config: Qwen3Config, layer_idx: int
) -> tuple[int | None, bool]:
    """Resolve ``(sliding_window, causal)`` for one DFlash draft layer.

    +----------------------+-------------------------+--------------------------------+
    | Config               | ``layer_type``          | *``causal``                    |
    +======================+=========================+================================+
    | ``layer_types``      | SWA if ``use_swa``      | True if ``layer_types[i]=SWA`` |
    |                      | else ``layer_types[i]`` | else False                     |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | SWA                     | False                          |
    | + ``use_swa=True``   |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | Full                    | False                          |
    | + ``use_swa=False``  |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    * If ``dflash_config.causal`` is set, its value overrides ``causal`` for all layers.

    This is to support a varied ecosystem of checkpoints, including:
    - XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash (sets "use_swa", assumes non-causal)
    - z-lab/gemma-4-31B-it-DFlash (has mixed layer types, assumes causal only for SWA)
    - z-lab/Qwen3.5-9B-DFlash ("standard" DFlash, all full attn, assumes non-causal)
    """
    dflash_config = getattr(config, "dflash_config", None) or {}
    layer_types = getattr(config, "layer_types", None)
    use_swa = dflash_config.get("use_swa", False)

    any_sliding = False
    if layer_types is not None:
        num_sliding = sum(lt == _SLIDING_ATTENTION for lt in layer_types)
        any_sliding = num_sliding > 0
        # Mixed sliding/full attention needs multiple KV groups (V2 runner only).
        if (
            0 < num_sliding < len(layer_types)
            and not get_current_vllm_config().use_v2_model_runner
        ):
            raise NotImplementedError(
                "DFlash drafters with mixed sliding/full attention require "
                "the V2 model runner; relaunch with "
                "VLLM_USE_V2_MODEL_RUNNER=1."
            )

    # ``use_swa`` forces SWA on every layer, even an all-full ``layer_types``.
    if layer_types is None or (use_swa and not any_sliding):
        is_sliding = use_swa
    else:
        is_sliding = layer_types[layer_idx] == _SLIDING_ATTENTION

    sliding_window = None
    if is_sliding:
        sliding_window = dflash_config.get(
            "swa_window_size", getattr(config, "sliding_window", None)
        )
        if sliding_window is None:
            raise ValueError(
                "DFlash sliding attention requires a window size configured in "
                "dflash_config.swa_window_size or the top-level sliding_window."
            )

    return sliding_window, _dflash_layer_causal(config, layer_idx)


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        sliding_window: int | None = None,
        causal: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        self.sliding_window = sliding_window
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            sinks=self.attention_sink_bias,
        )
        self.causal = causal
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        # DFlash drafts store the sink-bias flag inside dflash_config; fall back
        # to the top-level attribute used by other (e.g. MiMo) configs.
        dflash_config = getattr(config, "dflash_config", None) or {}
        add_swa_attention_sink_bias = dflash_config.get(
            "attention_sink_bias",
            getattr(config, "add_swa_attention_sink_bias", False),
        )

        # Resolve this layer's attention mode (full vs sliding window, causal vs
        # non-causal) from the draft config.
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            add_swa_attention_sink_bias=add_swa_attention_sink_bias,
            sliding_window=sliding_window,
            causal=causal,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        # Compute dtype for the fused-KV path (dequant / scaled_mm).
        self.compute_dtype = getattr(
            vllm_config.model_config, "dtype", torch.bfloat16
        )

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Masked query slots are fed to the draft as `mask_token_id`. Most DFlash
        # checkpoints will have the mask embedding in the vocabulary embedding table
        # at that slot id. Some checkpoints (XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash) ship
        # with a separate mask embedding tensor to use instead. When present, we load it
        # and substitute it for embed_tokens[mask_token_id] when computing embeddings.
        self.mask_token_id = drafter_config.get("mask_token_id")
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            self.fc = ReplicatedLinear(
                input_size=_get_dflash_fc_input_size(
                    vllm_config,
                ),
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            # Replace masked slots with the dedicated mask embedding.
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        self._hidden_norm_weight = self.hidden_norm.weight.data

        projections = [a.qkv_proj for a in layers_attn]
        self._kv_strategy = _decide_context_kv_strategy(projections)
        self._fused_kv_bias: torch.Tensor | None = None

        if self._kv_strategy == ContextKVStrategy.FUSED:
            # Unquantized: original fused fast path (zero change).
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
            if has_bias:
                self._fused_kv_bias = torch.cat(
                    [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
                )
        elif self._kv_strategy == ContextKVStrategy.FUSED_DEQUANT:
            # Dequant to the compute dtype, keep the fused F.linear. The buffers
            # are built lazily on first use (after ``process_weights_after_loading``),
            # so quantized weights are in their final layout.
            kv_weights = [
                get_and_maybe_dequant_weights(p, out_dtype=self.compute_dtype)[
                    a.q_size :
                ]
                for p, a in zip(projections, layers_attn)
            ]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
            if has_bias:
                self._fused_kv_bias = torch.cat(
                    [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
                )
        elif self._kv_strategy == ContextKVStrategy.SCALED_MM:
            # Keep the quantized weights. After ``process_weights_after_loading``
            # the cutlass-fp8 path stores the weight in [K, N] layout, so slice
            # the K/V output COLUMNS and concatenate along N -> one fused W8A8 GEMM.
            self._fused_kv_weight = torch.cat(
                [p.weight[:, a.q_size :] for p, a in zip(projections, layers_attn)],
                dim=1,
            )
            self._fused_kv_scale = torch.cat(
                [
                    _kv_scale_vector(p, a.q_size, a.kv_size)
                    for p, a in zip(projections, layers_attn)
                ],
                dim=0,
            ).unsqueeze(0)  # [1, L * 2 * kv_size] per-column
            if has_bias:
                self._fused_kv_bias = torch.cat(
                    [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
                )
        else:  # PER_LAYER
            self._kv_projections = [(a.qkv_proj, a.q_size) for a in layers_attn]

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
        # tensor so the per-layer K-norm runs as a single grouped kernel.
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._build_context_kv_buffers(layers_attn, has_bias)

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- KV projection (one fused GEMM for all layers when possible) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )

        if self._kv_strategy == ContextKVStrategy.SCALED_MM:
            all_kv_flat = self._project_context_kv_scaled(normed_context_states)
        elif self._kv_strategy == ContextKVStrategy.PER_LAYER:
            all_kv_flat = self._project_context_kv_per_layer(normed_context_states)
        else:  # FUSED / FUSED_DEQUANT
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )

        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v

    def _project_context_kv_scaled(self, normed: torch.Tensor) -> torch.Tensor:
        """Fused W8A8 projection: dynamic FP8 activation + one cutlass_scaled_mm.

        Weight is stored in [K, N] (cutlass layout); ``_fused_kv_scale`` is the
        per-output-column (K/V row) scale vector."""
        x_q, scale_a = ops.scaled_fp8_quant(
            normed, scale=None, use_per_token_if_dynamic=True
        )
        out_dtype = (
            normed.dtype
            if normed.dtype in (torch.bfloat16, torch.float16)
            else torch.bfloat16
        )
        return ops.cutlass_scaled_mm(
            x_q,
            self._fused_kv_weight,
            scale_a,
            self._fused_kv_scale,
            out_dtype,
            self._fused_kv_bias,
        )

    def _project_context_kv_per_layer(self, normed: torch.Tensor) -> torch.Tensor:
        """Per-layer quantized projection; output layout matches the fused path
        (``[num_ctx, L * 2 * kv_size]``).

        Q rows are projected and discarded (packed kernels cannot compute only
        the K/V rows); each layer's bias is applied in its module.
        """
        return _project_kv_per_layer(normed, self._kv_projections)

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        # --- Grouped RMSNorm K across all layers ([L, num_ctx, nkv, hd]) ---
        # The weight is selected per layer by the outermost (layer) index.
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            # Build the fused-KV buffers on first use.  This runs after the
            # loader has called ``process_weights_after_loading``, so quantized
            # weights are in their final layout.
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        all_k, all_v = self._project_context_kv(context_states, num_ctx, L, nkv, hd)
        all_k_normed = self._normalize_context_k(all_k)

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(L):
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue  # dummy run: skip cache ops
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _preprocess(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for name, loaded_weight in weights:
            if "attention_sink_bias" in name:
                # Sink bias is per-head; shard it across TP ranks like the
                # attention heads themselves.
                heads_per_rank = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank
                )
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._preprocess(weights), mapper=self.hf_to_vllm_mapper
        )


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        """Per-layer attention causality, aligned with
        get_draft_kv_cache_layer_names."""
        return [layer.self_attn.causal for layer in self.model.layers]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlash drafter expects {expected} concatenated aux hidden "
                f"features but received {hidden_states.shape[-1]}. This usually "
                "means the draft model's target_layer_ids reference layers that "
                "do not exist in the target model (incompatible draft/target pair)."
            )
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash embeds masked slots via mask_token_id (optionally "
                "overridden by a mask_embedding.pt file); it should not ship a "
                "mask_hidden weight."
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # Route the separately-trained mask embedding (if shipped) through the
        # standard weight loader alongside the rest of the draft weights.
        mask_embedding = self._read_mask_embedding()
        if mask_embedding is not None:
            model_weights["model.mask_embedding"] = mask_embedding
            self.model.has_separate_mask_embedding = True

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.model.has_separate_mask_embedding:
            skip_substrs.append("mask_embedding")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        # NOTE: fused-KV buffers are intentionally NOT built here.  They are
        # built lazily on first use in ``precompute_and_store_context_kv``,
        # after the loader has run ``process_weights_after_loading`` on every
        # layer, so that ``get_and_maybe_dequant_weights`` sees quantized
        # weights in their final layout (this is required for the fused-KV
        # path to support quantized drafters).

    def _read_mask_embedding(self) -> torch.Tensor | None:
        """Checks for an override mask embedding in `mask_embedding.pt` and returns it.

        Some checkpoints ship a separately-trained mask embedding for the mask token,
        which we use to overwrite the embedding for `mask_token_id`. This helper
        checks for the file, loads the pytorch tensor, and returns the embedding to use.

        Returns None if the override file is not present.
        """
        mask_token_id = self.model.mask_token_id
        if mask_token_id is None:
            return None

        MASK_EMBEDDING_FILENAME = "mask_embedding.pt"
        data = get_hf_file_bytes(
            MASK_EMBEDDING_FILENAME,
            self.draft_model_config.model,
            self.draft_model_config.revision,
        )
        if data is None:
            return None

        state = torch.load(io.BytesIO(data), weights_only=True)
        if isinstance(state, dict):
            if state.get("mask_token_id", mask_token_id) != mask_token_id:
                raise ValueError(
                    f"{MASK_EMBEDDING_FILENAME} mask_token_id does not match "
                    f"dflash_config.mask_token_id ({mask_token_id}). "
                    f"Got {state.get('mask_token_id')}."
                )
            state = state["embedding"]

        logger.info(
            "Loaded DFlash mask embedding for mask_token_id %s from %s",
            mask_token_id,
            MASK_EMBEDDING_FILENAME,
        )
        return state.reshape(-1)
