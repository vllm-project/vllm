# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch RWKV-7 "Goose" model.

Attention-free recurrent LM. Per layer, per sequence the only state is a
matrix-valued WKV state of shape ``(num_heads, head_dim, head_v_dim)`` plus a
1-token shift buffer. There is no KV cache, so memory is constant in
context length.

Reference: https://arxiv.org/abs/2503.14456 (RWKV-7 "Goose").
Public weights: ``fla-hub/rwkv7-{0.1B,0.4B,1.5B,2.9B}-{g1,world}``.

Inference uses a vendored copy of fla's ``fused_mul_recurrent_rwkv7`` kernel
(see ``vllm/model_executor/layers/fla/ops/rwkv7/``). The chunked prefill
kernel is a planned follow-up; the recurrent kernel handles all sequence
lengths but is slower for long prefills.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.rwkv7_mixer import (
    RWKV7Attention,
    compute_token_shift_delta,
    default_a_lora,
    default_decay_lora,
    default_gate_lora,
    default_v_lora,
    resolve_state_slots,
    scatter_intermediate_shift_blocks,
    update_shift_state,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.rwkv7_attn import Rwkv7AttentionMetadata

from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _resolve_lora_dim(
    config_value: int | None,
    default_fn,
    *args,
) -> int:
    return default_fn(*args) if config_value is None else int(config_value)


def _resolve_value_dim(
    config_value_dim,
    layer_idx: int,
    hidden_size: int,
) -> int:
    """Pick the per-layer value_dim, falling back to hidden_size."""
    if config_value_dim is None:
        return hidden_size
    if isinstance(config_value_dim, (list, tuple)):
        return int(config_value_dim[layer_idx])
    return int(config_value_dim)


def _resolve_num_heads(
    config_num_heads: int | None,
    hidden_size: int,
    head_dim: int,
) -> int:
    # Mirror fla's RWKV7Attention precedence: when head_dim is set, num_heads
    # is always derived from `hidden_size // head_dim`, even if the config
    # also lists a num_heads field. The configured value is informational —
    # several public checkpoints (e.g. fla-hub/rwkv7-0.1B-g1) ship a
    # num_heads value that disagrees with hidden_size/head_dim, and fla
    # silently uses the head_dim-derived value.
    if head_dim is not None:
        assert hidden_size % head_dim == 0, (
            f"hidden_size ({hidden_size}) must be divisible by head_dim ({head_dim})"
        )
        return hidden_size // head_dim
    if config_num_heads is None:
        raise ValueError("Either head_dim or num_heads must be set")
    return int(config_num_heads)


class RWKV7FeedForward(nn.Module):
    """Channel-mix FFN with cu_seqlens-aware token-shift, ReLU² activation.

    Per RWKV-7, the FFN has its own token-shift (separate from the time-mix
    attn block). The shift state cache lives on the parent ``RWKV7Attention``
    mixer at ``mixer.kv_cache[1]`` (slot 0 is the attn shift; slot 1 is the
    FFN shift; slot 2 is the recurrent matrix state).

    Like the time-mix block, the entire FFN body (state I/O + projections)
    runs inside a single ``torch.ops.vllm.rwkv7_channel_mix`` custom op so
    the model's ``@support_torch_compile`` decorator treats the FFN as
    opaque and does not trace stale ``mixer.kv_cache`` pointers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        attn_module: "RWKV7Attention",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attn_module = attn_module
        self.prefix = prefix

        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        torch.ops.vllm.rwkv7_channel_mix(hidden_states, output, self.prefix)

    def _full_forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return  # profile run

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.attn_module.prefix]
        assert isinstance(attn_metadata, Rwkv7AttentionMetadata)

        m = attn_metadata
        num_actual_tokens = m.num_actual_tokens
        num_decodes = m.num_decodes
        num_prefills = m.num_prefills
        num_decode_tokens = m.num_decode_tokens
        is_all = m.is_mamba_cache_all
        x = hidden_states[:num_actual_tokens]

        ffn_shift_cache = self.attn_module.kv_cache[1]

        if m.read_slot is None or m.write_slot is None:
            read_slot, write_slot = resolve_state_slots(
                m.state_indices_tensor,
                m.block_idx_last_computed_token,
                m.block_idx_last_scheduled_token,
            )
        else:
            read_slot, write_slot = m.read_slot, m.write_slot
        delta = compute_token_shift_delta(
            hidden_states=x,
            query_start_loc=m.query_start_loc,
            state_indices=read_slot,
            shift_state_cache=ffn_shift_cache,
            has_initial_state=m.has_initial_state,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )

        mixed = x.addcmul(delta, self.x_k)
        output[:num_actual_tokens] = self.value(torch.relu(self.key(mixed)).pow(2))

        update_shift_state(
            hidden_states=x,
            query_start_loc=m.query_start_loc,
            state_indices=write_slot,
            shift_state_cache=ffn_shift_cache,
        )
        if is_all and num_prefills > 0:
            cu_seqlens = m.query_start_loc.to(torch.int32)
            cu_seqlens_p_long = (cu_seqlens[num_decodes:] - cu_seqlens[num_decodes]).to(
                torch.long
            )
            scatter_intermediate_shift_blocks(
                hidden_states=x,
                query_start_loc_p=cu_seqlens_p_long,
                state_indices_tensor_p=m.state_indices_tensor[num_decodes:],
                block_idx_first_scheduled_p=m.block_idx_first_scheduled_token[
                    num_decodes:
                ],
                block_idx_last_scheduled_p=m.block_idx_last_scheduled_token[
                    num_decodes:
                ],
                num_computed_tokens_p=m.num_computed_tokens_p,
                block_size=m.mamba_block_size,
                shift_state_cache=ffn_shift_cache,
                num_decode_tokens=num_decode_tokens,
            )


class RWKV7DecoderLayer(nn.Module):
    """One transformer-style block: attn_norm → mixer → ffn_norm → ffn."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        norm_eps = getattr(config, "norm_eps", 1e-5)
        norm_bias = getattr(config, "norm_bias", True)
        norm_first = getattr(config, "norm_first", True)
        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", 64)
        num_heads = _resolve_num_heads(
            getattr(config, "num_heads", None), hidden_size, head_dim
        )
        value_dim = _resolve_value_dim(
            getattr(config, "value_dim", None), layer_idx, hidden_size
        )
        decay_lora = _resolve_lora_dim(
            getattr(config, "decay_low_rank_dim", None),
            default_decay_lora,
            hidden_size,
            head_dim,
        )
        a_lora = _resolve_lora_dim(
            getattr(config, "a_low_rank_dim", None),
            default_a_lora,
            hidden_size,
            head_dim,
        )
        v_lora = _resolve_lora_dim(
            getattr(config, "v_low_rank_dim", None),
            default_v_lora,
            hidden_size,
            head_dim,
        )
        gate_lora = _resolve_lora_dim(
            getattr(config, "gate_low_rank_dim", None),
            default_gate_lora,
            hidden_size,
        )

        # The leading-layer pre_norm only exists when norm_first=True (RWKV-7
        # default). It normalizes the embeddings before they enter layer 0.
        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=norm_bias)

        self.attn_norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=norm_bias)
        self.attn = RWKV7Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_lora,
            gate_low_rank_dim=gate_lora,
            a_low_rank_dim=a_lora,
            v_low_rank_dim=v_lora,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
            value_dim=value_dim,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=norm_bias)
        intermediate_size = getattr(
            config,
            "intermediate_size",
            32 * (((hidden_size * 4) + 31) // 32),
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            attn_module=self.attn,
            prefix=f"{prefix}.ffn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block. ``v_first`` is mutated in place at layer 0."""
        # Layer 0 only: extra LayerNorm of the embeddings before the block.
        if hasattr(self, "pre_norm"):
            hidden_states = self.pre_norm(hidden_states)

        residual = hidden_states
        x = self.attn_norm(hidden_states)
        attn_out = torch.zeros_like(x)
        # Custom op: mutates attn_out, mutates v_first at layer 0.
        self.attn(x, attn_out, v_first)
        hidden_states = residual + attn_out

        residual = hidden_states
        x = self.ffn_norm(hidden_states)
        ffn_out = torch.zeros_like(x)
        self.ffn(x, ffn_out)
        hidden_states = residual + ffn_out

        return hidden_states


@support_torch_compile
class RWKV7Model(nn.Module):
    """RWKV-7 backbone. Embeddings → N x DecoderLayer (with v_first thread) → norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RWKV7DecoderLayer(
                config=config,
                vllm_config=vllm_config,
                layer_idx=int(prefix.rsplit(".", 1)[-1]),
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        norm_eps = getattr(config, "norm_eps", 1e-5)
        norm_bias = getattr(config, "norm_bias", True)
        self.norm = nn.LayerNorm(config.hidden_size, eps=norm_eps, bias=norm_bias)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "v_first"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

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
            v_first = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            v_first = intermediate_tensors["v_first"]

        for layer in self.layers:
            hidden_states = layer(hidden_states, v_first)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "v_first": v_first}
            )

        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Skip extra biases that GPTQ/AWQ checkpoints sometimes ship.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                # Layer 0 has no v_lora; checkpoints that do ship one
                # (rare) should be ignored rather than rejected.
                if ".v_lora." in name and ".layers.0." in name:
                    continue
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class RWKV7ForCausalLM(
    nn.Module, HasInnerState, IsAttentionFree, SupportsMambaPrefixCaching
):
    """RWKV-7 LM head wrapper.

    No KV cache. Cache machinery from ``HasInnerState`` plumbs in the per-
    sequence (shift_state, recurrent_state) tuple. ``IsAttentionFree``
    tells the engine not to allocate paged attention storage.

    Speculative decoding, prefix caching, and pipeline parallelism are
    not yet supported and are hard-asserted off.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.rwkv7_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        cfg = vllm_config.model_config.hf_config
        head_dim = getattr(cfg, "head_dim", 64)
        num_heads = _resolve_num_heads(
            getattr(cfg, "num_heads", None), cfg.hidden_size, head_dim
        )
        # head_v_dim follows hidden_size for the dense Goose family. If a
        # checkpoint ships a per-layer value_dim list with mixed values,
        # the cache spec uses the first layer's value as a representative
        # — uniform value_dim is enforced at runtime in the mixer.
        value_dim = _resolve_value_dim(
            getattr(cfg, "value_dim", None), 0, cfg.hidden_size
        )
        head_v_dim = value_dim // num_heads

        return MambaStateShapeCalculator.rwkv7_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            hidden_size=cfg.hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            head_v_dim=head_v_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.rwkv7_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        super().__init__()
        # v1 limitations — surface them clearly at construction time so we
        # don't fail mid-forward with confusing kernel asserts.
        assert vllm_config.parallel_config.pipeline_parallel_size == 1, (
            "RWKV-7 does not yet support pipeline parallelism"
        )
        # Speculative decoding is wired up at the kernel and mixer level
        # but cross-round state alignment is not yet correct in production
        # (subtle token-duplication on repetitive prompts, kv-cache allocator
        # asserts on multi-request workloads). Keep this gate in place until
        # those are resolved.
        assert vllm_config.speculative_config is None, (
            "RWKV-7 does not yet support speculative decoding"
        )

        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = RWKV7Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.lm_head.tie_weights(self.model.embeddings)

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
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
