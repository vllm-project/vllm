# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import (
    CacheConfig,
    ModelConfig,
    get_current_vllm_config,
)
from vllm.distributed import divide
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.fla.ops.rwkv7 import (
    chunk_rwkv7,
    fused_mul_recurrent_rwkv7,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.rwkv7_attn import Rwkv7AttentionMetadata


class LoRA(nn.Module):
    """Low-rank projection used by RWKV-7's w/a/v/g gates.

    Mirrors fla's `fla.layers.rwkv6.LoRA`: down → activation → up, with an
    optional bias added inside the up-projection. Parameter names
    (``lora.0.weight``, ``lora.2.weight``, ``lora.2.bias``) match fla so
    that ``fla-hub/rwkv7-*`` checkpoints load without remapping.

    TP behavior: input is the full (replicated) hidden state, so the down
    projection is replicated. Output shards along its head/channel axis,
    so the up projection is column-parallel and returns the local rank's
    slice without cross-rank gather.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = True,
        activation: str | None = "tanh",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.act: nn.Module = nn.Identity()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation `{activation}`.")

        # ``lora.0`` matches the fla checkpoint layout (down, then activation,
        # then up). The down projection sees the full hidden input on every
        # rank, so it stays replicated; the up projection's output is sharded
        # along the head/channel axis to feed downstream per-rank ops.
        self.lora = nn.ModuleDict(
            {
                "0": ReplicatedLinear(
                    input_dim,
                    low_rank_dim,
                    bias=False,
                    prefix=f"{prefix}.lora.0",
                ),
                "2": ColumnParallelLinear(
                    low_rank_dim,
                    output_dim,
                    bias=bias,
                    gather_output=False,
                    prefix=f"{prefix}.lora.2",
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lora["0"](x)
        h = self.act(h)
        out, _ = self.lora["2"](h)
        return out


def compute_token_shift_delta(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    shift_state_cache: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    num_decodes: int,
    num_prefills: int,
) -> torch.Tensor:
    """Compute ``delta = shifted - hidden_states`` for the packed batch.

    For each sequence the "shifted" tensor is::

        [cached_prev_token, x_0, x_1, ..., x_{T-2}]

    where ``cached_prev_token`` is the previous step's last hidden state for
    the sequence (or zeros for a freshly-started prefill).

    Args:
        hidden_states: ``[T_total, D]``.
        query_start_loc: ``[N+1]`` cumulative token offsets per sequence.
        state_indices: ``[N]`` block ids in ``shift_state_cache``.
        shift_state_cache: ``[num_blocks, 1, D]``.
        has_initial_state: ``[num_prefills]`` bool, or ``None``. ``True`` means
            the corresponding prefill sequence has cached state (resumed
            prefill). Decode sequences (the first ``num_decodes`` slots) are
            assumed to always have valid cached state.
    """
    T_total, D = hidden_states.shape
    prev = hidden_states.new_zeros(T_total, D)
    if T_total > 1:
        prev[1:] = hidden_states[:-1]

    seq_starts = query_start_loc[:-1].to(torch.long)
    cached_prev = shift_state_cache[state_indices, 0]  # [N, D]
    if has_initial_state is not None and num_prefills > 0:
        mask = has_initial_state.to(cached_prev.dtype).unsqueeze(-1)
        cached_prev_prefill = cached_prev[num_decodes:] * mask
        cached_prev = torch.cat([cached_prev[:num_decodes], cached_prev_prefill], dim=0)
    prev[seq_starts] = cached_prev.to(prev.dtype)
    return prev - hidden_states


def update_shift_state(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    shift_state_cache: torch.Tensor,
) -> None:
    """Write the last token of each sequence into ``shift_state_cache``."""
    last_token_indices = query_start_loc[1:].to(torch.long) - 1
    last_tokens = hidden_states[last_token_indices]  # [N, D]
    shift_state_cache[state_indices, 0] = last_tokens.to(shift_state_cache.dtype)


def resolve_state_slots(
    state_indices_tensor: torch.Tensor,
    block_idx_last_computed: torch.Tensor | None,
    block_idx_last_scheduled: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the cache-slot indices for read (last computed) and write (last
    scheduled) under either default or 'all' mamba_cache_mode.

    Default mode: ``state_indices_tensor`` is 1D, both slots = same value.
    'all' mode: ``state_indices_tensor`` is 2D ``[N, max_blocks]``; gather the
    column at ``block_idx_*`` per sequence.
    """
    if block_idx_last_computed is None or block_idx_last_scheduled is None:
        return state_indices_tensor, state_indices_tensor
    read_idx = state_indices_tensor.gather(
        1, block_idx_last_computed.long().unsqueeze(1)
    ).squeeze(1)
    write_idx = state_indices_tensor.gather(
        1, block_idx_last_scheduled.long().unsqueeze(1)
    ).squeeze(1)
    return read_idx, write_idx


def scatter_intermediate_shift_blocks(
    hidden_states: torch.Tensor,
    query_start_loc_p: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    block_idx_first_scheduled_p: torch.Tensor,
    block_idx_last_scheduled_p: torch.Tensor,
    num_computed_tokens_p: torch.Tensor,
    block_size: int,
    shift_state_cache: torch.Tensor,
    num_decode_tokens: int,
) -> None:
    """Snapshot the end-of-block tokens of each prefill into shift cache.

    For prefill sequence ``i`` with ``first_scheduled=f``, ``last_scheduled=l``,
    we write ``hidden_states`` at the absolute position
    ``(k+1)*block_size - 1`` to ``shift_state_cache[state_indices_p[i, k]]``
    for every fully-completed block ``k in [f, l)``. The last block ``l`` is
    always written by ``update_shift_state`` separately (it may be partial).

    Args:
        hidden_states: ``[T_total, D]`` (the layer input).
        query_start_loc_p: cumulative offsets within the prefill chunk,
            shape ``[num_prefills + 1]``.
        state_indices_tensor_p: ``[num_prefills, max_blocks]`` cache slot ids.
        block_idx_first_scheduled_p, block_idx_last_scheduled_p: ``[num_prefills]``
        num_computed_tokens_p: ``[num_prefills]`` tokens already cached
            entering this step (used to align to the prefill input).
        block_size: cache block size in tokens (= kernel chunk_size for
            chunk_stride=1, which is what we use).
        shift_state_cache: ``[num_blocks, 1, D]``
        num_decode_tokens: prefill input starts at this offset in
            ``hidden_states``.
    """
    num_prefills = block_idx_first_scheduled_p.shape[0]
    if num_prefills == 0:
        return

    starts_p = query_start_loc_p[:-1].to(torch.long)
    for i in range(num_prefills):
        f = int(block_idx_first_scheduled_p[i].item())
        last = int(block_idx_last_scheduled_p[i].item())
        if last <= f:
            continue
        num_computed = int(num_computed_tokens_p[i].item())
        seq_start_in_prefill = int(starts_p[i].item())
        # Absolute end-of-block positions in the full sequence:
        # (k+1)*block_size - 1
        # Position within the prefill input: that minus num_computed
        # Position in the global packed tensor:
        # + num_decode_tokens + seq_start_in_prefill
        for k in range(f, last):
            end_of_block = (k + 1) * block_size - 1
            pos_in_prefill = end_of_block - num_computed
            if pos_in_prefill < 0:
                continue  # block already fully cached, shouldn't happen
            global_idx = num_decode_tokens + seq_start_in_prefill + pos_in_prefill
            slot = int(state_indices_tensor_p[i, k].item())
            shift_state_cache[slot, 0] = hidden_states[global_idx].to(
                shift_state_cache.dtype
            )


def scatter_intermediate_recurrent_blocks(
    varlen_states: torch.Tensor,
    chunk_offsets_p: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    block_idx_first_scheduled_p: torch.Tensor,
    block_idx_last_scheduled_p: torch.Tensor,
    num_computed_tokens_p: torch.Tensor,
    block_size: int,
    chunk_size: int,
    recurrent_state_cache: torch.Tensor,
) -> None:
    """Scatter per-chunk-end recurrent states to intermediate cache blocks.

    For prefill sequence ``i`` with ``first_scheduled=f``, ``last_scheduled=l``,
    write the state at the END of each block ``k in [f, l)`` to
    ``recurrent_state_cache[state_indices_p[i, k]]``. Block ``l`` is written
    separately from ``final_state``.

    With ``chunk_stride = block_size // chunk_size``, the state at the end of
    block ``k`` corresponds to the chunk-state at index
    ``chunk_offsets_p[i] + (k - first_scheduled) * chunk_stride - 1`` (with an
    alignment offset when ``num_computed_tokens`` isn't block-aligned).
    """
    num_prefills = block_idx_first_scheduled_p.shape[0]
    if num_prefills == 0:
        return
    chunk_stride = block_size // chunk_size
    if chunk_stride < 1:
        return  # block_size must be >= chunk_size
    for i in range(num_prefills):
        f = int(block_idx_first_scheduled_p[i].item())
        last = int(block_idx_last_scheduled_p[i].item())
        if last <= f:
            continue
        n_blocks_to_fill = last - f
        # First chunk index of this prefill in the global varlen_states tensor.
        first_chunk = int(chunk_offsets_p[i].item())
        first_aligned_chunk = first_chunk + chunk_stride - 1
        num_unaligned_computed = int(num_computed_tokens_p[i].item()) % block_size
        if num_unaligned_computed > 0:
            first_aligned_chunk -= num_unaligned_computed // chunk_size
        idx_start = first_aligned_chunk
        idx_step = chunk_stride
        idx_end = idx_start + n_blocks_to_fill * idx_step
        states_to_write = varlen_states[idx_start:idx_end:idx_step]
        cache_slots = state_indices_tensor_p[i, f:last]
        recurrent_state_cache[cache_slots] = states_to_write.to(
            recurrent_state_cache.dtype
        )


def _gate_output_correction(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """RWKV-7 post-attention residual + gate.

    Equivalent to fla's Triton-fused `gate_output_correction`. Used because
    the fused kernel adds significant vendoring weight for negligible
    inference speedup at typical decode batch sizes.

    Shapes: o [T, H*D], r/k/v [T, H, D], r_k [H, D], g [T, H*D].
    """
    correction = ((r * k * r_k.unsqueeze(0)).sum(-1, keepdim=True) * v).reshape(o.shape)
    return (o + correction) * g


class RWKV7Attention(nn.Module, MambaBase):
    """RWKV-7 "Goose" time-mixer.

    Attention-free recurrent block. Per-sequence per-layer state:
      - shift_state: previous token's hidden state (for token_shift mix)
      - recurrent_state: matrix-valued WKV state of shape (H, K, V), fp32

    The model also threads a `v_first` value-residual through every layer:
    layer 0 sets `v_first = v`; later layers do
    `v ← lerp(v, v_first, sigmoid(v_lora(xv)))`. `v_first` is a Python
    local in the model loop — never cached, never crosses the custom op
    boundary.
    """

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.RWKV7_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype, torch.dtype]:  # type: ignore[override]
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.rwkv7_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(  # type: ignore[override]
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
        return MambaStateShapeCalculator.rwkv7_state_shape(
            tp_world_size=self.tp_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            head_v_dim=self.head_v_dim,
        )

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        decay_low_rank_dim: int,
        gate_low_rank_dim: int,
        a_low_rank_dim: int,
        v_low_rank_dim: int,
        norm_eps: float,
        layer_idx: int,
        value_dim: int | None = None,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "rwkv7_attn",
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.hidden_size = hidden_size
        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.norm_eps = norm_eps
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config

        # Per-rank head/channel counts. ``divide`` raises if not evenly
        # divisible, so the model surfaces a clear error instead of silently
        # mis-sharding when ``num_heads`` is not a multiple of ``tp_size``.
        self.num_heads_per_tp = divide(num_heads, self.tp_size)
        self.key_dim_per_tp = self.num_heads_per_tp * head_dim
        self.value_dim_per_tp = self.num_heads_per_tp * self.head_v_dim

        # Time-shift mix scalars. Multiplied with the layer input (full hidden,
        # replicated across ranks), so they stay replicated.
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Per-channel and per-(head, head_dim) scalar gates. These multiply
        # post-projection tensors that are already sharded along the head
        # axis, so the parameters shard the same way.
        self.k_k = nn.Parameter(torch.zeros(self.key_dim_per_tp))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim_per_tp))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads_per_tp, head_dim))
        for p in (self.k_k, self.k_a):
            set_weight_attrs(p, {"weight_loader": sharded_weight_loader(0)})
        # ``r_k`` shards along the head axis (dim 0).
        set_weight_attrs(self.r_k, {"weight_loader": sharded_weight_loader(0)})

        # Full projections (no bias). At TP=1 these are just regular Linears.
        self.r_proj = ColumnParallelLinear(
            hidden_size,
            self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.r_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.value_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # LoRAs. Layer 0 has no v_lora (it sets v_first = v directly).
        # Up-projection outputs feed downstream per-rank ops (kernel inputs,
        # gate-output correction), so the up shards along head/channel axis
        # via ColumnParallelLinear inside ``LoRA``.
        self.w_lora = LoRA(
            hidden_size,
            self.key_dim,
            low_rank_dim=decay_low_rank_dim,
            activation="tanh",
            prefix=f"{prefix}.w_lora",
        )
        if self.layer_idx != 0:
            self.v_lora = LoRA(
                hidden_size,
                self.value_dim,
                low_rank_dim=v_low_rank_dim,
                activation=None,
                prefix=f"{prefix}.v_lora",
            )
        self.a_lora = LoRA(
            hidden_size,
            self.key_dim,
            low_rank_dim=a_low_rank_dim,
            activation=None,
            prefix=f"{prefix}.a_lora",
        )
        # g_lora has sigmoid between the two legs and no bias on the up-proj
        self.g_lora = LoRA(
            hidden_size,
            self.value_dim,
            low_rank_dim=gate_low_rank_dim,
            bias=False,
            activation="sigmoid",
            prefix=f"{prefix}.g_lora",
        )

        # GroupNorm operates on each rank's local head shard. The affine
        # weight/bias are loaded sharded along channel axis 0.
        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads_per_tp,
            num_channels=self.value_dim_per_tp,
            eps=self.head_dim * norm_eps,
            affine=True,
        )
        if self.tp_size > 1:
            for p in (self.g_norm.weight, self.g_norm.bias):
                set_weight_attrs(p, {"weight_loader": sharded_weight_loader(0)})

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        v_first: torch.Tensor,
    ) -> None:
        """Apply the RWKV-7 mixer.

        Both ``output`` and ``v_first`` are mutated in place. ``v_first`` is
        written at layer 0 (with this layer's ``v`` projection) and read but
        not written at later layers.

        The whole body executes inside a single ``torch.ops.vllm.rwkv7_attention``
        custom op so that ``@support_torch_compile`` on the model class
        treats the mixer as opaque. This is required because the state I/O
        in ``_full_forward`` reads ``self.kv_cache`` whose buffers are
        allocated after model construction — tracing those reads under
        Dynamo captures stale tensor pointers.
        """
        torch.ops.vllm.rwkv7_attention(
            hidden_states,
            output,
            v_first,
            self.prefix,
        )

    # ------------------------------------------------------------------
    # Custom-op body
    # ------------------------------------------------------------------
    def _full_forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        v_first: torch.Tensor,
    ) -> None:
        """Run the entire mixer block including all state I/O.

        Mutates ``output[:num_actual_tokens]`` and, at layer 0, mutates
        ``v_first[:num_actual_tokens]`` with the freshly computed ``v``.
        """
        forward_context: ForwardContext = get_forward_context()
        ctx_meta: object = forward_context.attn_metadata
        if ctx_meta is None:
            return  # profile/dummy run

        if isinstance(ctx_meta, dict):
            ctx_meta = ctx_meta[self.prefix]
        assert isinstance(ctx_meta, Rwkv7AttentionMetadata)
        m: Rwkv7AttentionMetadata = ctx_meta
        num_actual_tokens = m.num_actual_tokens
        num_decodes = m.num_decodes
        num_prefills = m.num_prefills
        num_decode_tokens = m.num_decode_tokens
        state_indices = m.state_indices_tensor
        query_start_loc = m.query_start_loc
        assert state_indices is not None and query_start_loc is not None
        cu_seqlens = query_start_loc.to(torch.int32)
        is_all = m.is_mamba_cache_all

        # kv_cache layout (3-tuple, set up in get_state_shape):
        #   [0] attn shift state  : [num_blocks, 1, D]
        #   [1] ffn  shift state  : [num_blocks, 1, D]   — written by the FFN
        #   [2] recurrent matrix  : [num_blocks, H, K, V]
        attn_shift_cache = self.kv_cache[0]
        ffn_shift_cache = self.kv_cache[1]
        recurrent_state_cache = self.kv_cache[2]

        # Resolved cache-slot ids per active sequence:
        #   - read_slot[i]  = slot to load initial state from (last computed)
        #   - write_slot[i] = slot to store the per-sequence final state to
        # These come from metadata so decode CUDA graph capture sees stable
        # tensor addresses even when prefix-cache hits change the slots.
        if m.read_slot is None or m.write_slot is None:
            read_slot, write_slot = resolve_state_slots(
                state_indices,
                m.block_idx_last_computed_token,
                m.block_idx_last_scheduled_token,
            )
        else:
            read_slot, write_slot = m.read_slot, m.write_slot
        # Zero the read slot for freshly-started prefill sequences.
        if num_prefills > 0 and m.has_initial_state is not None:
            fresh_mask = ~m.has_initial_state
            if fresh_mask.any():
                fresh_read = read_slot[num_decodes:][fresh_mask]
                attn_shift_cache[fresh_read] = 0
                ffn_shift_cache[fresh_read] = 0
                recurrent_state_cache[fresh_read] = 0

        x = hidden_states[:num_actual_tokens]

        delta = compute_token_shift_delta(
            hidden_states=x,
            query_start_loc=query_start_loc,
            state_indices=read_slot,
            shift_state_cache=attn_shift_cache,
            has_initial_state=m.has_initial_state,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )

        x_r = self.x_r.view(-1)
        x_w = self.x_w.view(-1)
        x_k = self.x_k.view(-1)
        x_v = self.x_v.view(-1)
        x_a = self.x_a.view(-1)
        x_g = self.x_g.view(-1)
        xr = torch.addcmul(x, delta, x_r)
        xw = torch.addcmul(x, delta, x_w)
        xk = torch.addcmul(x, delta, x_k)
        xv = torch.addcmul(x, delta, x_v)
        xa = torch.addcmul(x, delta, x_a)
        xg = torch.addcmul(x, delta, x_g)

        r, _ = self.r_proj(xr)
        k, _ = self.k_proj(xk)
        v, _ = self.v_proj(xv)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        if self.layer_idx == 0:
            v_first[:num_actual_tokens] = v
        else:
            v = torch.lerp(v, v_first[:num_actual_tokens], self.v_lora(xv).sigmoid())

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = (k * self.k_k).view(
            num_actual_tokens, self.num_heads_per_tp, self.head_dim
        )
        kk = F.normalize(kk, dim=-1, p=2.0).reshape(num_actual_tokens, -1)
        k = k.addcmul(k * (a - 1), self.k_a)

        # [1, T_total, H_per_tp, D] reshape for the packed-batch kernel
        r_4d = r.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_dim
        ).contiguous()
        w_4d = w.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_dim
        ).contiguous()
        k_4d = k.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_dim
        ).contiguous()
        v_4d = v.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_v_dim
        ).contiguous()
        kk_4d = kk.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_dim
        ).contiguous()
        a_4d = a.view(
            1, num_actual_tokens, self.num_heads_per_tp, self.head_dim
        ).contiguous()

        initial_state = recurrent_state_cache[read_slot]
        if initial_state.dtype != torch.float32:
            initial_state = initial_state.to(torch.float32)

        # Dispatch:
        #  - pure-decode batch (every active sequence is 1 token): use the
        #    recurrent kernel; cheaper than chunk for T=1 and the only path
        #    captured by cudagraphs (UNIFORM_SINGLE_TOKEN_DECODE).
        #  - any prefill present: use the chunked kernel which processes
        #    chunk_size tokens in parallel per program. Decodes within a
        #    mixed batch ride along as length-1 chunks — cheap.
        if num_prefills == 0:
            core_out, final_state = fused_mul_recurrent_rwkv7(
                r=r_4d,
                w=w_4d,
                k=k_4d,
                v=v_4d,
                kk=kk_4d,
                a=a_4d,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
            varlen_states = None
        else:
            # chunk_rwkv7 takes (a=-kk, b=kk*a_gate) — fla's parameterization
            a_chunk = -kk_4d
            b_chunk = kk_4d * a_4d
            core_out, final_state, varlen_states = chunk_rwkv7(
                r=r_4d,
                w=w_4d,
                k=k_4d,
                v=v_4d,
                a=a_chunk,
                b=b_chunk,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                # Our log-decay is `-0.6065 * sigmoid(...)`, which sits in
                # (-0.607, 0] — well inside the safe-gate range.
                safe_gate=True,
                chunk_size=64,
                return_varlen_states=is_all,
            )
        # Write per-sequence final state to the last_scheduled cache slot.
        recurrent_state_cache[write_slot] = final_state.to(recurrent_state_cache.dtype)

        # Under prefix-caching ('all') mode, also fill the intermediate blocks
        # with the per-chunk-end recurrent states produced by chunk_rwkv7.
        if is_all and num_prefills > 0 and varlen_states is not None:
            assert m.block_idx_first_scheduled_token is not None
            assert m.block_idx_last_scheduled_token is not None
            assert m.mamba_block_size is not None
            # Indices of the prefill rows in the [N, max_blocks] state table.
            state_indices_p = state_indices[num_decodes:]
            # Per-prefill chunk offsets in the global varlen_states tensor.
            # cu_seqlens for the prefill subset:
            cu_seqlens_p = cu_seqlens[num_decodes:] - cu_seqlens[num_decodes]
            cu_seqlens_p = cu_seqlens_p.to(torch.long)
            chunk_size = 64
            # Number of chunks per sequence = cdiv(seq_len, chunk_size).
            # varlen_states is laid out over the whole packed batch, including
            # decode chunks before the prefill region.
            seq_lens_all = cu_seqlens[1:].to(torch.long) - cu_seqlens[:-1].to(
                torch.long
            )
            decode_chunk_offset = (
                (seq_lens_all[:num_decodes] + chunk_size - 1) // chunk_size
            ).sum()
            seq_lens_p = cu_seqlens_p[1:] - cu_seqlens_p[:-1]
            chunks_per_seq = (seq_lens_p + chunk_size - 1) // chunk_size
            # Prefix sum to get the global start index of each prefill's chunks.
            chunk_offsets_p = torch.zeros(
                num_prefills + 1, dtype=torch.long, device=cu_seqlens.device
            )
            chunk_offsets_p[0] = decode_chunk_offset
            chunk_offsets_p[1:] = decode_chunk_offset + chunks_per_seq.cumsum(0)

            scatter_intermediate_recurrent_blocks(
                varlen_states=varlen_states,
                chunk_offsets_p=chunk_offsets_p,
                state_indices_tensor_p=state_indices_p,
                block_idx_first_scheduled_p=m.block_idx_first_scheduled_token[
                    num_decodes:
                ],
                block_idx_last_scheduled_p=m.block_idx_last_scheduled_token[
                    num_decodes:
                ],
                num_computed_tokens_p=m.num_computed_tokens_p,
                block_size=m.mamba_block_size,
                chunk_size=chunk_size,
                recurrent_state_cache=recurrent_state_cache,
            )

        o = self.g_norm(core_out.view(num_actual_tokens, self.value_dim_per_tp))
        o = _gate_output_correction(
            o,
            r_4d.view(num_actual_tokens, self.num_heads_per_tp, self.head_dim),
            k_4d.view(num_actual_tokens, self.num_heads_per_tp, self.head_dim),
            self.r_k,
            v_4d.view(num_actual_tokens, self.num_heads_per_tp, self.head_v_dim),
            g,
        )
        proj, _ = self.o_proj(o)
        output[:num_actual_tokens] = proj

        update_shift_state(
            hidden_states=x,
            query_start_loc=query_start_loc,
            state_indices=write_slot,
            shift_state_cache=attn_shift_cache,
        )
        if is_all and num_prefills > 0:
            assert m.block_idx_first_scheduled_token is not None
            assert m.block_idx_last_scheduled_token is not None
            assert m.mamba_block_size is not None
            cu_seqlens_p_long = (cu_seqlens[num_decodes:] - cu_seqlens[num_decodes]).to(
                torch.long
            )
            scatter_intermediate_shift_blocks(
                hidden_states=x,
                query_start_loc_p=cu_seqlens_p_long,
                state_indices_tensor_p=state_indices[num_decodes:],
                block_idx_first_scheduled_p=m.block_idx_first_scheduled_token[
                    num_decodes:
                ],
                block_idx_last_scheduled_p=m.block_idx_last_scheduled_token[
                    num_decodes:
                ],
                num_computed_tokens_p=m.num_computed_tokens_p,
                block_size=m.mamba_block_size,
                shift_state_cache=attn_shift_cache,
                num_decode_tokens=num_decode_tokens,
            )


def rwkv7_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    v_first: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self: RWKV7Attention = forward_context.no_compile_layers[layer_name]
    self._full_forward(hidden_states=hidden_states, output=output, v_first=v_first)


def rwkv7_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    v_first: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="rwkv7_attention",
    op_func=rwkv7_attention,
    mutates_args=["output", "v_first"],
    fake_impl=rwkv7_attention_fake,
)


def rwkv7_channel_mix(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op wrapping the RWKV-7 channel-mix (FFN) block.

    Same compile-opacity rationale as the time-mix op: state I/O on
    ``self.kv_cache[1]`` (the FFN shift cache, owned by the parent mixer)
    must not be traced by Dynamo.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._full_forward(hidden_states=hidden_states, output=output)


def rwkv7_channel_mix_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="rwkv7_channel_mix",
    op_func=rwkv7_channel_mix,
    mutates_args=["output"],
    fake_impl=rwkv7_channel_mix_fake,
)


def _round_to_32(value: float) -> int:
    return int(round(value / 32) * 32)


def default_decay_lora(hidden_size: int, head_dim: int) -> int:
    factor = head_dim / 64
    return max(32, _round_to_32(2.5 * (hidden_size**0.5) * factor))


def default_a_lora(hidden_size: int, head_dim: int) -> int:
    factor = head_dim / 64
    return max(32, _round_to_32(2.5 * (hidden_size**0.5) * factor))


def default_v_lora(hidden_size: int, head_dim: int) -> int:
    factor = head_dim / 64
    return max(32, _round_to_32(1.7 * (hidden_size**0.5) * factor))


def default_gate_lora(hidden_size: int) -> int:
    return max(32, _round_to_32(5.0 * (hidden_size**0.5)))
