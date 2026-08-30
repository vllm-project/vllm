# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Orthrus model compatible with HuggingFace weights."""

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Attention
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    maybe_prefix,
)

_BUILDING_DIFFUSION_DRAFT: ContextVar[bool] = ContextVar(
    "orthrus_building_diffusion_draft", default=False
)


@contextmanager
def building_diffusion_draft():
    """Marks the enclosing model construction as Orthrus' diffusion draft.

    Set by ``OrthrusProposer._get_model`` around its ``get_model()`` call.
    Everything built inside this context (a) allocates the extra
    ``attn_diff`` paged attention layers and (b) routes ``forward()``
    through the diffusion path.

    An explicit context flag is used rather than inspecting the VllmConfig
    because the draft is constructed with the *target's* ``model_config``
    still on the config object -- ``_create_draft_vllm_config`` never
    replaces it and ``initialize_model`` passes ``vllm_config`` through
    unchanged, only using its separate ``model_config`` argument to pick
    the architecture. So identity checks like
    ``vllm_config.model_config is speculative_config.draft_model_config``
    are always False here and cannot distinguish draft from target.
    """
    token = _BUILDING_DIFFUSION_DRAFT.set(True)
    try:
        yield
    finally:
        _BUILDING_DIFFUSION_DRAFT.reset(token)


def _force_write_diffusion_kv(
    attn_diff: Attention, key: torch.Tensor, value: torch.Tensor
) -> None:
    """Write the diffusion block's own new key/value into the shared cache.

    ``attn_diff.kv_sharing_target_layer_name`` (set by
    ``OrthrusProposer._setup_orthrus_kv_sharing``) makes this layer *read*
    the target layer's paged KV cache, which is what we want. But vLLM's
    own ``Attention.forward`` also uses that same field to decide whether
    to *write* this layer's key/value into the cache at all -- it skips
    the write whenever a kv_sharing target is set (see
    ``vllm.model_executor.layers.attention.attention.unified_kv_cache_update``'s
    callers), on the assumption that a kv-sharing layer contributes no new
    keys of its own (true for the existing supported pattern, Gemma4 MTP's
    Q-only attention -- no k_proj/v_proj at all -- but not true here:
    Orthrus's diffusion attention has its own k_proj_diff/v_proj_diff and
    needs its output written into the *same physical* shared cache so the
    rest of the block can attend to it in the same forward).

    Calling vLLM's own ``unified_kv_cache_update`` -> ``do_kv_cache_update``
    -> ``reshape_and_cache_flash`` custom op directly here (bypassing only
    ``Attention.forward``'s skip) crashes with an opaque PyTorch AOTI
    tensor-handle error ("aoti_torch_get_size ... API call failed"),
    on both a real decode step and vLLM's dummy/profiling run -- this
    compiled kernel has apparently never been exercised against a
    kv-sharing-*aliased* cache tensor before (the one existing kv-sharing
    pattern, Gemma4 MTP, never calls it at all), and something about that
    combination breaks an internal assumption inside the precompiled
    extension. Rather than debug a prebuilt CUDA kernel's internals,
    write the same physical slots with plain PyTorch indexing instead.

    The physical layout was confirmed by inspection rather than assumed:
    for this checkpoint/backend, ``attn_diff.kv_cache`` is a
    ``[num_blocks, num_kv_heads, block_size, 2 * head_dim]``-shaped view
    (matching ``do_kv_cache_update``'s own comment) whose *storage* is
    actually contiguous in ``[num_blocks, block_size, num_kv_heads,
    2 * head_dim]`` order (confirmed via ``.stride()`` -- the block-size
    and head dims are transposed relative to a naively-contiguous view of
    the reported shape). The last dim holds key then value concatenated
    per head (``do_kv_cache_update`` splits it the same way via
    ``.split(head_size, dim=-1)``). Slot numbers already follow the
    standard ``physical_block_id * block_size + offset`` convention
    ``set_inputs_first_pass`` uses to build slot_mapping in the first
    place, so decomposing a slot back into ``(block_idx, offset)`` and
    indexing directly reproduces the same addressing the CUDA kernel
    implements, without going through it.
    """
    forward_context = get_forward_context()
    if forward_context.attn_metadata is None:
        # vLLM's startup dummy/profiling run only needs realistic shapes,
        # not an actually-populated cache.
        return
    slot_mapping = forward_context.slot_mapping.get(attn_diff.layer_name)
    if slot_mapping is None:
        return
    kv_cache = attn_diff.kv_cache
    num_blocks, num_kv_heads, block_size, two_head_dim = kv_cache.shape
    head_dim = two_head_dim // 2
    num_actual_tokens = slot_mapping.shape[0]
    block_idx = slot_mapping // block_size
    offset = slot_mapping % block_size
    key = key[:num_actual_tokens].view(num_actual_tokens, num_kv_heads, head_dim)
    value = value[:num_actual_tokens].view(num_actual_tokens, num_kv_heads, head_dim)
    kv_cache[block_idx, :, offset, :head_dim] = key
    kv_cache[block_idx, :, offset, head_dim:] = value


# Orthrus adds a parallel set of "*_diff" attention projections next to the
# Qwen3 ones. Order matters only for readability here: matching is anchored on
# the full module segment (see resolve_stacked_weight_name), so "q_proj" does
# not also capture "q_proj_diff".
STACKED_PARAMS_MAPPING: list[tuple[str, str, str | int]] = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("qkv_proj_diff", "q_proj_diff", "q"),
    ("qkv_proj_diff", "k_proj_diff", "k"),
    ("qkv_proj_diff", "v_proj_diff", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


def resolve_stacked_weight_name(name: str) -> tuple[str, str | int] | None:
    """Maps a checkpoint weight name onto its fused vLLM parameter.

    Args:
        name: Checkpoint weight name, e.g. ``...self_attn.q_proj_diff.weight``.

    Returns:
        ``(vllm_param_name, shard_id)``, or ``None`` if ``name`` is not part
        of a fused parameter and should be loaded directly.
    """
    for param_name, weight_name, shard_id in STACKED_PARAMS_MAPPING:
        if f"{weight_name}." in name:
            return name.replace(weight_name, param_name), shard_id
    return None


class OrthrusAttention(Qwen3Attention):
    """Qwen3 attention with Orthrus diffusion-path projection weights.

    The standard ``forward`` method intentionally uses the autoregressive Qwen3
    path. Diffusion-mode generation needs additional scheduler and attention
    metadata support before it can be exposed safely in vLLM.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )
        self.qkv_proj_diff = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_diff",
        )
        self.o_proj_diff = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_diff",
        )
        self.q_norm_diff = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm_diff = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # EXPERIMENTAL (milestone 3, unvalidated against vLLM's real
        # scheduler/CI): a second paged Attention layer for the diffusion
        # path. Built ONLY for the speculative-decode *draft* instance --
        # OrthrusProposer.load_model wires each of these to KV-share with
        # the corresponding same-index *target* layer's self.attn, so a
        # diffusion-block forward reads the target's real, already-populated
        # cache and needs no cache of its own.
        #
        # It must not be built on the target: a layer with no
        # kv_sharing_target_layer_name gets a full KVCacheSpec of its own
        # (see GPUModelRunner.get_kv_cache_spec), so building it there would
        # allocate a second, entirely unused set of KV cache -- roughly
        # halving usable context/concurrency for every Orthrus load,
        # including plain AR serving with no speculative_config at all.
        if _BUILDING_DIFFUSION_DRAFT.get():
            self.attn_diff = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_diff",
            )

    def forward_diffusion_paged(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion-path forward through vLLM's real paged attention.

        EXPERIMENTAL / milestone 3: unlike ``forward_diffusion`` (which takes
        explicit offline KV tensors, used for milestone-1/2 validation),
        this goes through ``self.attn_diff`` -- a normal vLLM Attention
        layer -- so it participates in the live scheduler's KV cache and
        forward context like any other decode step. It is only meaningful
        when ``self.attn_diff.kv_sharing_target_layer_name`` has been wired
        (by ``OrthrusProposer.load_model``) to the corresponding *target*
        model layer's ``self.attn``, so this reads the target's real cache.

        Force-writes this block's own diffusion key/value into the shared
        cache before reading it back (see ``_force_write_diffusion_kv``):
        setting ``kv_sharing_target_layer_name`` makes vLLM's own
        ``Attention.forward`` skip the cache *write* for this layer
        entirely (see ``unified_kv_cache_update``'s callers) -- that skip
        exists for kv-sharing patterns with no new keys to write at all
        (e.g. Gemma4 MTP's Q-only attention), which reads correctly from
        the target's pre-existing cache but is wrong here: without this,
        every position in the proposed block could only ever see the AR
        history from *before* this round, never any other position's own
        embedding, regardless of the attention mask used -- measured as
        acceptance rate staying flat regardless of block length or mask.
        """
        qkv, _ = self.qkv_proj_diff(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm_diff(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm_diff(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        _force_write_diffusion_kv(self.attn_diff, k, v)
        attn_output = self.attn_diff(q, k, v)
        output, _ = self.o_proj_diff(attn_output)
        return output

    def forward_diffusion(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion-mode forward for a block of mask-token queries.

        Attends over this layer's AR-path cached key/value states (already
        computed by a prior autoregressive forward) plus this block's own
        diffusion keys/values, with no causal restriction. This matches the
        reference implementation's inference-time behavior: the
        block-diagonal masking in ``generate_dual_pass_mask`` is only
        exercised during training on multiple packed diffusion blocks -- a
        single served diffusion block simply attends over the fixed AR
        prefix plus itself (see ``OrthrusModel.forward`` in the reference
        ``modeling_orthrus.py``, where ``causal_mask``/``flex_block_mask``
        are only built when both ``self.training`` and ``is_diffusion_pass``
        are true).

        Args:
            positions: ``[block_len]`` absolute sequence positions.
            hidden_states: ``[block_len, hidden_size]``.
            cached_key: ``[ar_seq_len, num_kv_heads, head_dim]`` AR-path
                cached keys for this layer, already RoPE-applied.
            cached_value: ``[ar_seq_len, num_kv_heads, head_dim]`` AR-path
                cached values for this layer.

        Returns:
            ``[block_len, hidden_size]`` attention output for the block.
        """
        block_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj_diff(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm_diff(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm_diff(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(block_len, self.num_heads, self.head_dim)
        k = k.view(block_len, self.num_kv_heads, self.head_dim)
        v = v.view(block_len, self.num_kv_heads, self.head_dim)

        full_key = torch.cat([cached_key, k], dim=0)
        full_value = torch.cat([cached_value, v], dim=0)

        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            full_key = full_key.repeat_interleave(num_groups, dim=1)
            full_value = full_value.repeat_interleave(num_groups, dim=1)

        # [1, heads, seq, head_dim] for scaled_dot_product_attention.
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = full_key.transpose(0, 1).unsqueeze(0)
        v_t = full_value.transpose(0, 1).unsqueeze(0)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, is_causal=False, scale=self.scaling
        ).squeeze(0)
        attn_output = attn_output.transpose(0, 1).reshape(block_len, -1)

        output, _ = self.o_proj_diff(attn_output)
        return output

    def forward_ar_eager(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_key: torch.Tensor | None,
        cached_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager (non-paged) autoregressive forward.

        Used by the standalone diffusion-decode demo loop
        (``OrthrusForCausalLM.generate_with_diffusion``), which drives this
        model outside vLLM's paged-attention runtime so it can also serve
        as the AR verification pass in the propose/verify/accept loop.
        Mirrors the normal ``forward`` math (inherited from
        ``Qwen3Attention``) but explicitly manages a plain tensor KV cache
        instead of going through ``self.attn``.

        Args:
            positions: ``[block_len]`` absolute sequence positions.
            hidden_states: ``[block_len, hidden_size]``.
            cached_key: ``[past_len, num_kv_heads, head_dim]`` or ``None``
                for the first (prefill) call.
            cached_value: matching ``cached_key``.

        Returns:
            ``(output, updated_key_cache, updated_value_cache)``.
        """
        block_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(block_len, self.num_heads, self.head_dim)
        k = k.view(block_len, self.num_kv_heads, self.head_dim)
        v = v.view(block_len, self.num_kv_heads, self.head_dim)

        if cached_key is not None:
            full_key = torch.cat([cached_key, k], dim=0)
            full_value = torch.cat([cached_value, v], dim=0)
        else:
            full_key, full_value = k, v

        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            attn_key = full_key.repeat_interleave(num_groups, dim=1)
            attn_value = full_value.repeat_interleave(num_groups, dim=1)
        else:
            attn_key, attn_value = full_key, full_value

        total_kv_len = attn_key.shape[0]
        past_len = total_kv_len - block_len
        q_idx = torch.arange(block_len, device=hidden_states.device).unsqueeze(1)
        kv_idx = torch.arange(total_kv_len, device=hidden_states.device).unsqueeze(0)
        causal = kv_idx <= (q_idx + past_len)

        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = attn_key.transpose(0, 1).unsqueeze(0)
        v_t = attn_value.transpose(0, 1).unsqueeze(0)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=causal, scale=self.scaling
        ).squeeze(0)
        attn_output = attn_output.transpose(0, 1).reshape(block_len, -1)

        output, _ = self.o_proj(attn_output)
        return output, full_key, full_value


class OrthrusDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        self.self_attn = OrthrusAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
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

    def forward_diffusion(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn.forward_diffusion(
            positions=positions,
            hidden_states=hidden_states,
            cached_key=cached_key,
            cached_value=cached_value,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_ar_eager(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cached_key: torch.Tensor | None,
        cached_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, new_key, new_value = self.self_attn.forward_ar_eager(
            positions=positions,
            hidden_states=hidden_states,
            cached_key=cached_key,
            cached_value=cached_value,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_key, new_value

    def forward_diffusion_paged(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EXPERIMENTAL (milestone 3): paged-attention diffusion forward."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn.forward_diffusion_paged(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class OrthrusModel(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=OrthrusDecoderLayer,
        )
        # Read inside building_diffusion_draft()'s context span, same as
        # OrthrusForCausalLM.is_orthrus_diffusion_draft -- see forward's
        # docstring for why the dispatch needs to live here rather than in
        # the outer LM's forward.
        self.is_orthrus_diffusion_draft = _BUILDING_DIFFUSION_DRAFT.get()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Dispatch to the diffusion path from *inside* the compiled entry
        point, not from the caller.

        ``@support_torch_compile`` instruments this class's ``__call__``,
        not any other method -- calling ``forward_diffusion_paged``
        directly (as this used to, from ``OrthrusForCausalLM.forward``)
        bypasses torch.compile and CUDA graph capture for the whole
        diffusion path entirely, which is why
        ``OrthrusProposer.initialize_cudagraph_keys`` has to force
        ``cudagraph_mode=NONE`` for the draft. Branching here instead
        means both paths go through the same ``__call__``-wrapped entry
        point that compilation/capture actually instruments.
        """
        if self.is_orthrus_diffusion_draft:
            return self.forward_diffusion_paged(input_ids, positions, inputs_embeds)
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def forward_diffusion(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        per_layer_cached_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Runs a diffusion-mode forward for a block of mask tokens.

        This proposes ``len(input_ids)`` future tokens in a single forward
        pass by attending each layer's diffusion projections over that
        layer's already-computed AR cache, instead of the usual sequential
        one-token-per-step autoregressive path. Candidates produced from
        this pass must still be verified against a normal AR forward before
        being accepted (see the reference implementation's ``generate``
        method for the accept/reject loop that makes this lossless).

        Args:
            input_ids: ``[block_len]`` token ids for the diffusion block --
                the first position holds the last accepted AR token and the
                rest are ``mask_token_id``, matching the reference
                implementation's block construction.
            positions: ``[block_len]`` absolute sequence positions.
            per_layer_cached_kv: one ``(key, value)`` pair per decoder layer,
                each shaped ``[ar_seq_len, num_kv_heads, head_dim]``.

        Returns:
            ``[block_len, hidden_size]`` final hidden states for the block.
        """
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer, (cached_key, cached_value) in zip(self.layers, per_layer_cached_kv):
            hidden_states, residual = layer.forward_diffusion(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                cached_key=cached_key,
                cached_value=cached_value,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_diffusion_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """EXPERIMENTAL (milestone 3): paged-attention diffusion forward.

        Used when this ``OrthrusModel`` is loaded as a speculative-decode
        *draft* (see ``OrthrusProposer``): each layer's diffusion attention
        reads the corresponding *target* model layer's live paged KV cache
        via ``kv_sharing_target_layer_name`` (wired by
        ``OrthrusProposer.load_model``), instead of the offline tensors
        ``forward_diffusion`` takes for milestone-1/2 validation.
        """
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.forward_diffusion_paged(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_ar_eager(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        per_layer_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Eager (non-paged) autoregressive forward across all layers.

        Companion to ``forward_diffusion`` for the standalone demo loop in
        ``OrthrusForCausalLM.generate_with_diffusion`` -- used both for the
        very first prefill and for the AR verification pass over a
        proposed diffusion block.

        Args:
            input_ids: ``[block_len]`` token ids.
            positions: ``[block_len]`` absolute sequence positions.
            per_layer_cache: one ``(key, value)`` pair per layer (or
                ``None`` per layer on the first prefill call).

        Returns:
            ``(hidden_states, updated_per_layer_cache)``, where
            ``hidden_states`` is ``[block_len, hidden_size]`` and each
            updated cache entry is ``[past_len + block_len, num_kv_heads,
            head_dim]``.
        """
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, cached in zip(self.layers, per_layer_cache):
            cached_key, cached_value = cached if cached is not None else (None, None)
            hidden_states, residual, new_key, new_value = layer.forward_ar_eager(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                cached_key=cached_key,
                cached_value=cached_value,
            )
            new_cache.append((new_key, new_value))
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, new_cache

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            resolved = resolve_stacked_weight_name(name)
            if resolved is not None:
                name, shard_id = resolved
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class OrthrusForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "qkv_proj_diff": [
            "q_proj_diff",
            "k_proj_diff",
            "v_proj_diff",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = OrthrusModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
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
        # Always go through self.model's __call__ (not a direct method call)
        # so torch.compile/CUDA-graph capture sees this invocation -- see
        # OrthrusModel.forward's docstring for why the diffusion branch used
        # to bypass compilation entirely by being dispatched from here.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def propose_diffusion_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        per_layer_cached_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Proposes a block of future tokens via Orthrus' diffusion path.

        Returns ``[block_len, vocab_size]`` logits. Candidates sampled from
        these logits are not yet verified against the AR path -- callers
        must run an AR forward over the proposed block and accept/reject
        per-position (matching the reference implementation's exact
        intra-model consistency check) before treating them as generated
        output.
        """
        hidden_states = self.model.forward_diffusion(
            input_ids, positions, per_layer_cached_kv
        )
        return self.compute_logits(hidden_states)

    @torch.inference_mode()
    def generate_with_diffusion(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Standalone, single-request greedy diffusion-mode decode loop.

        Ports the reference implementation's ``OrthrusLM.generate`` propose
        /verify/accept loop (greedy-only) onto this module's own eager
        forward paths, bypassing vLLM's paged-attention runtime and
        scheduler entirely. This exists to validate that Orthrus' diffusion
        decoding actually works end-to-end against vLLM's own weight-loaded
        model code; it is not wired into vLLM's continuous-batching engine
        (KV-cache paging, CUDA graphs, and multi-request batching are all
        future work -- see the PR discussion for what that would require).

        Args:
            input_ids: ``[1, prompt_len]`` prompt token ids.
            max_new_tokens: number of new tokens to generate, upper bound.
            eos_token_id: stop generation once this token is produced.

        Returns:
            ``[1, prompt_len + generated_len]`` full token sequence.
        """
        assert input_ids.shape[0] == 1, (
            "generate_with_diffusion only supports batch size 1"
        )
        device = input_ids.device
        block_size = self.config.block_size
        mask_token_id = self.config.mask_token_id
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )

        prompt_ids = input_ids[0]
        num_input_tokens = prompt_ids.shape[0]
        max_length = num_input_tokens + max_new_tokens

        output_ids = torch.full(
            (max_length + block_size,), mask_token_id, dtype=torch.long, device=device
        )
        output_ids[:num_input_tokens] = prompt_ids

        # Initial AR prefill.
        positions = torch.arange(num_input_tokens, device=device)
        num_layers = len(self.model.layers)
        hidden_states, kv_cache = self.model.forward_ar_eager(
            prompt_ids, positions, [None] * num_layers
        )
        logits = self.compute_logits(hidden_states)
        next_token = logits[-1].argmax(dim=-1)

        start_idx = num_input_tokens
        output_ids[start_idx] = next_token
        if next_token.item() == eos_token_id:
            return output_ids[: start_idx + 1].unsqueeze(0)

        while start_idx < max_length - 1:
            diff_len = min(block_size, max_length - start_idx)
            diff_block_ids = torch.full(
                (diff_len,), mask_token_id, dtype=torch.long, device=device
            )
            diff_block_ids[0] = output_ids[start_idx]
            diff_positions = torch.arange(
                start_idx, start_idx + diff_len, device=device
            )

            # --- Propose: one diffusion forward predicts the whole block. ---
            if diff_len > 1:
                diff_logits = self.propose_diffusion_block(
                    diff_block_ids, diff_positions, kv_cache
                )
                diff_tokens = diff_logits[:-1].argmax(dim=-1)
            else:
                diff_tokens = torch.empty((0,), dtype=torch.long, device=device)

            proposed_block = torch.cat(
                [output_ids[start_idx : start_idx + 1], diff_tokens]
            )

            # --- Verify: a normal AR forward over the proposed block must
            # agree, position by position, for the block to be accepted
            # losslessly (this AR pass is exactly what would run anyway
            # for a plain non-diffusion decode of these positions). ---
            ar_hidden, verify_cache = self.model.forward_ar_eager(
                proposed_block, diff_positions, kv_cache
            )
            ar_logits = self.compute_logits(ar_hidden)
            ar_tokens = ar_logits.argmax(dim=-1)

            matches = diff_tokens == ar_tokens[:-1]
            acceptance_len = (
                int(matches.cumprod(dim=0).sum().item()) if diff_tokens.numel() else 0
            )
            next_token = ar_tokens[acceptance_len]

            end_idx = start_idx + acceptance_len + 1
            accepted_block = proposed_block[: acceptance_len + 1]

            eos_positions = (accepted_block == eos_token_id).nonzero()
            if len(eos_positions) > 0:
                eos_offset = int(eos_positions[0, 0].item())
                output_ids[start_idx : start_idx + eos_offset + 1] = accepted_block[
                    : eos_offset + 1
                ]
                return output_ids[: start_idx + eos_offset + 1].unsqueeze(0)

            output_ids[start_idx:end_idx] = accepted_block
            # The AR verification pass already computed correct KV entries
            # for every accepted position (and only those) -- keep just the
            # accepted prefix of the newly computed cache.
            keep_len = accepted_block.shape[0]  # == acceptance_len + 1
            kv_cache = [
                (
                    k[: k.shape[0] - diff_len + keep_len],
                    v[: v.shape[0] - diff_len + keep_len],
                )
                for k, v in verify_cache
            ]
            start_idx = end_idx

            if start_idx < max_length:
                output_ids[start_idx] = next_token
                if next_token.item() == eos_token_id:
                    return output_ids[: start_idx + 1].unsqueeze(0)

        return output_ids[:max_length].unsqueeze(0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # AutoWeightsLoader's constructor no longer accepts skip_prefixes;
        # tied lm_head/embed_tokens weights are deduplicated internally by
        # qualname-aliasing (see AutoWeightsLoader's weight-tying handling).
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


OrthrusLM = OrthrusForCausalLM
