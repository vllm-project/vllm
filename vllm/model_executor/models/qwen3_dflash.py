# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


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
    config_causal = dflash_config.get("causal", None)

    SLIDING_ATTENTION = "sliding_attention"
    any_sliding = False
    if layer_types is not None:
        num_sliding = sum(lt == SLIDING_ATTENTION for lt in layer_types)
        any_sliding = num_sliding > 0
        all_sliding = num_sliding == len(layer_types)
        if any_sliding and not all_sliding:
            # Mixed sliding/full attention needs per-layer causal metadata and
            # multiple KV-cache groups, which DFlash does not yet support.
            raise NotImplementedError(
                "DFlash does not yet support mixed sliding/full attention via "
                "layer_types; see "
                "https://github.com/vllm-project/vllm/issues/40898."
            )

    default_causal = False
    if layer_types is None or (use_swa and not any_sliding):
        # An absent ``layer_types`` (or the all-"full_attention" one that may
        # be synthesized when the checkpoint omits it) must not override
        # ``dflash_config.use_swa``, which forces SWA on every layer.
        is_sliding = use_swa
    else:
        is_sliding = layer_types[layer_idx] == SLIDING_ATTENTION
        # Full-attention layers default non-causal; SWA layers default causal.
        default_causal = is_sliding

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

    causal = config_causal if config_causal is not None else default_causal
    return sliding_window, causal


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
        # NOTE: `causal` is currently unused here, but will be needed in the future
        # to support models with different causality per-layer.
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
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
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

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

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
        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
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
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            if "attention_sink_bias" in name:
                if name not in params_dict:
                    continue
                # Sink bias is per-head; shard it across TP ranks like the
                # attention heads themselves.
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DominoHead(nn.Module):
    def __init__(
        self, *, config, vllm_config: VllmConfig, quant_config, prefix: str = ""
    ):
        super().__init__()
        dflash_config = getattr(config, "dflash_config", {}) or {}

        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.emb_dim = int(dflash_config["emb_dim"])

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )

        self.embed_proj = nn.Sequential(
            ReplicatedLinear(
                input_size=config.hidden_size + self.gru_hidden_dim,
                output_size=self.emb_dim,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_proj.0"),
                return_bias=False,
            ),
            nn.SiLU(),
            ReplicatedLinear(
                input_size=self.emb_dim,
                output_size=config.vocab_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_proj.2"),
                return_bias=False,
            ),
        )

    def init_state(self, embed_input_ids, prefix_token_ids: torch.Tensor):
        prefix_embeds = embed_input_ids(prefix_token_ids)
        _, gru_hidden = self.prefix_gru(prefix_embeds)
        return gru_hidden

    def advance_state(
        self,
        embed_input_ids,
        token_ids: torch.Tensor,
        gru_hidden: torch.Tensor,
    ):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)
        token_embeds = embed_input_ids(token_ids)
        _, gru_hidden = self.prefix_gru(token_embeds, gru_hidden)
        return gru_hidden

    def compute_logits(
        self,
        parallel_hidden: torch.Tensor,
        gru_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ):
        squeeze_time_dim = parallel_hidden.dim() == 2
        if squeeze_time_dim:
            parallel_hidden = parallel_hidden.unsqueeze(1)

        if base_logits.dim() == 2:
            base_logits = base_logits.unsqueeze(1)

        state = gru_hidden.transpose(0, 1)
        correction_input = torch.cat([parallel_hidden, state], dim=-1)
        correction_bias = self.embed_proj(correction_input)

        logits = base_logits + correction_bias
        if squeeze_time_dim:
            logits = logits.squeeze(1)
        return logits


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

        dflash_config = getattr(self.config, "dflash_config", {}) or {}
        self.projector_type = dflash_config.get("projector_type")
        self.is_domino = self.projector_type == "domino"

        if self.is_domino:
            self.domino_head = DominoHead(
                config=self.config,
                vllm_config=vllm_config,
                quant_config=self.model.quant_config,
                prefix=maybe_prefix(prefix, "domino_head"),
            )
        else:
            self.domino_head = None

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
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False

        # Domino-only weights should be loaded into self.domino_head, not into the
        # DFlash backbone.
        domino_weights = []

        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash embeds masked slots via mask_token_id (optionally "
                "overridden by a mask_embedding.pt file); it should not ship a "
                "mask_hidden weight."
            )

            if getattr(self, "is_domino", False) and (
                name.startswith("prefix_gru.") or name.startswith("embed_proj.")
            ):
                domino_weights.append((f"domino_head.{name}", loaded_weight))
                continue

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

        if getattr(self, "is_domino", False):
            loader = AutoWeightsLoader(self)
            loader.load_weights(domino_weights)

            domino_param_names = {
                name
                for name, _ in self.named_parameters()
                if name.startswith("domino_head.")
            }
            loaded_domino_names = {name for name, _ in domino_weights}
            missing_domino_names = domino_param_names - loaded_domino_names

            if missing_domino_names:
                raise RuntimeError(
                    "Domino weight loading is incomplete. Missing: "
                    f"{sorted(missing_domino_names)}"
                )

        self.model._build_fused_kv_buffers()

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

    def init_domino_state(self, prefix_token_ids: torch.Tensor) -> torch.Tensor:
        if self.domino_head is None:
            raise RuntimeError("Domino head is not enabled.")
        return self.domino_head.init_state(self.embed_input_ids, prefix_token_ids)

    def advance_domino_state(
        self,
        token_ids: torch.Tensor,
        gru_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.domino_head is None:
            raise RuntimeError("Domino head is not enabled.")
        return self.domino_head.advance_state(
            self.embed_input_ids,
            token_ids,
            gru_hidden,
        )

    def compute_domino_logits(
        self,
        parallel_hidden: torch.Tensor,
        gru_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.domino_head is None:
            raise RuntimeError("Domino head is not enabled.")
        return self.domino_head.compute_logits(
            parallel_hidden,
            gru_hidden,
            base_logits,
        )
