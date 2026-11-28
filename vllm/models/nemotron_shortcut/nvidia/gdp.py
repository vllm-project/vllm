# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated DeltaProduct (GDP) linear-attention mixer for Nemotron Shortcut.

GDP is a gated delta rule that takes `gdp_num_householder` delta steps per
token. It reuses the GDN attention backend for its metadata and FlashInfer's
GDP kernels: `gdp_prefill.chunk_gated_delta_product` for prefill and
`cake_gdp.cake_gated_delta_product_mtp` for decode and speculative verify.
With `--use-replayssm`, decode and speculative verify run on
`gdp_conv_replay_decode.gdp_conv_replay` instead (see `gdp_replay.py`).
"""

import bisect
import functools
import importlib

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    direct_register_custom_op,
    get_kv_cache_torch_dtype,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .gdp_replay import (
    FLUSH_MIN,
    GdpReplayStep,
    GdpReplayTrackers,
    is_prefix_view,
)


def gdp_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
    mamba_ssm_cache_dtype: MambaDType = "auto",
) -> tuple[torch.dtype, torch.dtype]:
    """The GDP kernels read and write the recurrent state in float32."""
    conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    if mamba_ssm_cache_dtype not in ("auto", "float32"):
        raise ValueError(
            "Gated DeltaProduct recurrent state supports only auto and "
            f"float32; got {mamba_ssm_cache_dtype}"
        )
    return (conv_state_dtype, torch.float32)


def gdp_state_shape(
    tp_world_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    num_householder: int,
    num_spec: int = 0,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """GDP takes `num_householder` delta steps per token.

    Keys and values are projected per householder, so the conv widens; the
    recurrent state is a rank-`num_householder` transition of the same matrix
    and keeps the gated delta net shape.
    """
    conv_dim = divide(
        head_k_dim * num_k_heads * (1 + num_householder)
        + head_v_dim * num_v_heads * num_householder,
        tp_world_size,
    )
    state_len = conv_kernel_size - 1 + num_spec
    conv_state_shape = (
        (conv_dim, state_len) if is_conv_state_dim_first() else (state_len, conv_dim)
    )
    temporal_state_shape = (
        divide(num_v_heads, tp_world_size),
        head_v_dim,
        head_k_dim,
    )
    return conv_state_shape, temporal_state_shape


# Conv rows FlashInfer's gdp_conv_replay was tuned with, per tokens per step T.
_GDP_REPLAY_CONV_ROWS = {1: 4, 2: 4, 3: 8, 4: 8, 5: 8, 6: 8, 7: 12, 8: 12}


def gdp_replay_conv_rows(tokens: int) -> int:
    """Conv ring rows for steps of up to `tokens` tokens (FlashInfer's
    `gdp_conv_replay_decode.conv_state_rows`)."""
    if tokens not in _GDP_REPLAY_CONV_ROWS:
        raise ValueError(f"GDP replay runs 1..8 tokens per step; got {tokens}")
    return _GDP_REPLAY_CONV_ROWS[tokens]


def gdp_replay_ring_slots(tokens: int) -> int:
    """K / U / g ring slots for steps of up to `tokens` tokens in the banked
    layout: two banks of 8 + T (FlashInfer's `gdp_conv_replay_decode.ring_slots`)."""
    if tokens not in _GDP_REPLAY_CONV_ROWS:
        raise ValueError(f"GDP replay runs 1..8 tokens per step; got {tokens}")
    return 2 * (8 + tokens)


def gdp_replay_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
    mamba_ssm_cache_dtype: MambaDType = "auto",
) -> tuple[torch.dtype, ...]:
    """Dtypes of the replay page, in `gdp_replay_state_shape` order: conv ring,
    checkpoint, K ring, U ring, g ring. The op is fastest with an FP16
    checkpoint, so "auto" means FP16."""
    conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    if mamba_ssm_cache_dtype == "auto":
        checkpoint_dtype = torch.float16
    elif mamba_ssm_cache_dtype in ("float16", "bfloat16", "float32"):
        checkpoint_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
    else:
        raise ValueError(
            "GDP replay checkpoint supports only auto, float16, bfloat16 and "
            f"float32; got {mamba_ssm_cache_dtype}"
        )
    return (
        conv_state_dtype,
        checkpoint_dtype,
        torch.float16,
        torch.float16,
        torch.float32,
    )


def gdp_replay_state_shape(
    tp_world_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    num_householder: int,
    num_spec: int = 0,
) -> tuple[
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
]:
    """The page of FlashInfer's `gdp_conv_replay`, one block per request:

    0. conv ring `(R_c, conv_dim)`, token-major; the op indexes it itself.
    1. checkpoint `(heads, head_v_dim, head_k_dim)`: the state at the last
       flush, not the current state.
    2. K ring `(heads, R, num_householder, head_k_dim)`.
    3. U ring `(heads, R, num_householder, head_v_dim)`.
    4. g ring `(heads, R)`.
    """
    if is_conv_state_dim_first():
        raise ValueError(
            "GDP replay needs the token-major (SD) conv state; unset "
            "VLLM_SSM_CONV_STATE_LAYOUT=DS"
        )
    conv_shape, temporal_shape = gdp_state_shape(
        tp_world_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        num_householder,
    )
    tokens = 1 + num_spec
    ring = gdp_replay_ring_slots(tokens)
    heads = temporal_shape[0]
    return (
        (gdp_replay_conv_rows(tokens), conv_shape[1]),
        temporal_shape,
        (heads, ring, num_householder, head_k_dim),
        (heads, ring, num_householder, head_v_dim),
        (heads, ring),
    )


# FlashInfer's GDP kernels: the module and the function the mixer calls.
_GDP_KERNELS = (
    ("flashinfer.gdp_prefill", "chunk_gated_delta_product"),
    ("flashinfer.cake_gdp", "cake_gated_delta_product_mtp"),
)


@functools.cache
def missing_gdp_kernels() -> tuple[str, ...]:
    """The FlashInfer GDP kernels that cannot be imported."""
    missing = []
    for module_name, name in _GDP_KERNELS:
        try:
            if hasattr(importlib.import_module(module_name), name):
                continue
        except Exception:  # ImportError, or a module failing to load
            pass
        missing.append(f"{module_name}.{name}")
    return tuple(missing)


@functools.cache
def gdp_replay_kernel_missing() -> bool:
    """Whether FlashInfer's `gdp_conv_replay` cannot be imported."""
    try:
        module = importlib.import_module("flashinfer.gdp_conv_replay_decode")
        return not hasattr(module, "gdp_conv_replay")
    except Exception:  # ImportError, or a module failing to load
        return True


def gdp_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked GDP prefill over packed sequences: the outputs and each
    sequence's final state. `key`, `value` and `beta` hold each token's
    Householder steps on dim 1 ([tokens, steps, heads, ...]); q/k are already
    L2-normalized; `decay` is the log-space decay."""
    from flashinfer.gdp_prefill import chunk_gated_delta_product

    # FlashInfer takes a token's steps as consecutive sub-token rows and the
    # decay in linear space, whose neutral value is 1.
    return chunk_gated_delta_product(
        q=query,
        k=key.flatten(0, 1),
        v=value.flatten(0, 1),
        g=decay.exp(),
        beta=beta.flatten(0, 1).float(),
        num_householder=key.size(1),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
    )


def _pad_spec_window(t: torch.Tensor, dst: torch.Tensor, size: int) -> torch.Tensor:
    """Scatter ragged spec-decode tokens into a rectangular per-request block."""
    return t.new_zeros((size, *t.shape[1:])).index_copy_(0, dst, t)


_SPEC_ROWS_LEAD = "_gdp_spec_rows_lead"


def _spec_rows_lead(metadata: GDNAttentionMetadata) -> bool:
    """Whether a mixed step's spec requests are its leading requests, so that
    their tokens are the step's first `num_spec_decode_tokens` rows and the
    other requests' tokens follow. Read from the host masks once per step."""
    lead = getattr(metadata, _SPEC_ROWS_LEAD, None)
    if lead is None:
        masks = metadata.spec_sequence_masks_cpu
        lead = masks is not None and bool(masks[: metadata.num_spec_decodes].all())
        setattr(metadata, _SPEC_ROWS_LEAD, lead)
    return lead


def _make_gdp_sharded_weight_loader(
    blocks: list[tuple[int, int]], tp_size: int, tp_rank: int
):
    """Shard a fused GDP projection by head rather than by flat column.

    `blocks` describes the checkpoint packing as `(rows, num_householder)` per
    block. Keys, values and betas are laid out householder-major, so a
    contiguous column shard would hand a rank a mixture of householder slots
    instead of a head subset; slice each householder separately so every rank
    gets the same heads from all of them.
    """
    for rows, householder in blocks:
        assert (rows // householder) % tp_size == 0, (
            "every GDP projection block must shard evenly across heads"
        )

    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        shards = []
        offset = 0
        for rows, householder in blocks:
            per_householder = rows // householder
            shard = per_householder // tp_size
            for j in range(householder):
                start = offset + j * per_householder + tp_rank * shard
                shards.append(loaded_weight[start : start + shard])
            offset += rows
        param.data.copy_(torch.cat(shards, dim=0).reshape(param.shape))

    return weight_loader


def install_gdp_in_proj_loaders(
    in_proj: nn.Module, blocks: list[tuple[int, int]], tp_size: int, tp_rank: int
) -> None:
    """Give the fused GDP projection's weight and block scales the head-wise loader.

    The default column-parallel loader would shard by flat column. A quantized
    checkpoint's block scales with one row per weight row (e.g. MXFP8
    `weight_scale` of shape [rows, K / 32]) follow the weight's row layout, so
    they need the same sharding; otherwise a rank pairs its weight rows with the
    wrong scales whenever `tp_size > 1`. Scales with coarser rows (e.g. FP8
    128x128 block scales) or no row dimension (per-tensor) are not row-aligned
    and keep their loader.
    """
    in_proj.weight.weight_loader = _make_gdp_sharded_weight_loader(
        blocks, tp_size, tp_rank
    )
    weight_rows = in_proj.weight.shape[0]
    for scale_name in ("weight_scale", "weight_scale_inv"):
        scale = getattr(in_proj, scale_name, None)
        if scale is not None and scale.dim() == 2 and scale.shape[0] == weight_rows:
            scale.weight_loader = _make_gdp_sharded_weight_loader(
                blocks, tp_size, tp_rank
            )


class NemotronShortcutGDPMixer(nn.Module, MambaBase):
    """Gated DeltaProduct linear attention.

    Every token takes `gdp_num_householder` delta-rule steps instead of one,
    which raises the rank of the state transition without widening the state:
    keys, values and betas are projected per householder, while the query, the
    decay and the output gate stay per token.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        gdp_replay_trackers: GdpReplayTrackers | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        self.activation = "silu"

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if config.gdp_num_groups != config.gdp_num_heads:
            raise NotImplementedError(
                "Nemotron Shortcut GDP expects gdp_num_groups == gdp_num_heads"
            )
        if missing := missing_gdp_kernels():
            raise RuntimeError(
                "The GDP mixer needs FlashInfer's GDP kernels; missing: "
                + ", ".join(missing)
            )

        self.num_heads = divide(config.gdp_num_heads, tp_size)
        self.num_groups = divide(config.gdp_num_groups, tp_size)
        self.head_dim = config.gdp_head_dim
        self.key_head_dim = config.gdp_key_head_dim
        self.num_householder = config.gdp_num_householder
        self.conv_kernel_size = config.gdp_conv_kernel
        # Every dimension below is this rank's share.
        self.value_dim = self.num_heads * self.head_dim
        self.key_dim = self.num_groups * self.key_head_dim
        self.conv_dim = (
            self.num_householder * self.value_dim
            + (self.num_householder + 1) * self.key_dim
        )

        n = self.num_householder
        value_dim = config.gdp_num_heads * self.head_dim
        key_dim = config.gdp_num_groups * self.key_head_dim
        # The checkpoint packs [gate, conv inputs, per-householder beta, decay],
        # with the conv inputs in turn packed [value, key, query].
        in_proj_blocks = [
            (value_dim, 1),
            (n * value_dim, n),
            (n * key_dim, n),
            (key_dim, 1),
            (n * config.gdp_num_heads, n),
            (config.gdp_num_heads, 1),
        ]
        conv_blocks = in_proj_blocks[1:4]
        projection_size = sum(rows for rows, _ in in_proj_blocks)

        self.in_proj = ColumnParallelLinear(
            config.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        install_gdp_in_proj_loaders(self.in_proj, in_proj_blocks, tp_size, tp_rank)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_gdp_sharded_weight_loader(
                    conv_blocks, tp_size, tp_rank
                )
            },
        )
        # fp32 like Mamba2's A: the GDP kernels read them in fp32, and a model-dtype
        # parameter would cost two cast kernels per layer on every decode step.
        # requires_grad=False: `.float()` now returns the parameter itself, and the
        # CuTe-DSL decode kernel cannot DLPack-export a tensor that requires grad.
        # The replay op reads them as one [A_log; dt_bias] tensor, so under replay
        # both are views of it and loading them fills it in place.
        # On GDP layers, --use-replayssm selects FlashInfer's GDP replay. The flag
        # has its own name: MambaBase.use_replayssm selects the shared ReplaySSM
        # trackers (Mamba-2, Gated DeltaNet), which GDP replay does not use.
        self.use_gdp_replayssm = cache_config is not None and cache_config.use_replayssm
        if self.use_gdp_replayssm and gdp_replay_kernel_missing():
            raise RuntimeError(
                "--use-replayssm on GDP layers needs FlashInfer's "
                "gdp_conv_replay_decode.gdp_conv_replay"
            )
        self.gate_params: torch.Tensor | None = None
        if self.use_gdp_replayssm:
            self.gate_params = torch.empty(2, self.num_heads, dtype=torch.float32)
            a_log, dt_bias = self.gate_params[0], self.gate_params[1]
        else:
            a_log = torch.empty(self.num_heads, dtype=torch.float32)
            dt_bias = torch.empty(self.num_heads, dtype=torch.float32)
        self.A_log = nn.Parameter(a_log, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias, requires_grad=False)
        # The model shares one tracker set across its GDP layers; a standalone
        # layer keeps its own.
        self.gdp_replay_trackers: GdpReplayTrackers | None = None
        if self.use_gdp_replayssm:
            self.gdp_replay_trackers = gdp_replay_trackers or GdpReplayTrackers()
        # The state views from the first bind, at the KV cache's full capacity.
        self.gdp_replay_capacity_views: tuple[torch.Tensor, ...] | None = None
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.g_clamp_value = getattr(config, "gdp_g_clamp_value", None)
        self.norm = RMSNormGated(
            self.value_dim,
            eps=config.gdp_norm_eps,
            group_size=self.value_dim // self.num_groups,
            norm_before_gate=False,
        )
        set_weight_attrs(self.norm.weight, {"weight_loader": sharded_weight_loader(0)})
        self.out_proj = RowParallelLinear(
            value_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        self.num_spec = vllm_config.num_speculative_tokens
        # The replay op compiles once per batch size, in warmup's FULL decode
        # graphs; eager steps pad up to one of those (`_replay_batches`).
        self.replay_capture_sizes = sorted(
            compilation_config.cudagraph_capture_sizes or []
        )
        self.replay_decode_requests = (
            compilation_config.decode_cudagraph_capture_request_sizes
        )
        self._replay_batch_cache: dict[int, list[int]] = {}

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if self.gdp_replay_trackers is None:
            return
        self.gdp_replay_trackers.bind(self.kv_cache[0])
        # The replay op compiles per pool shape, block count included. An
        # extensible KV cache binds the pools at capacity for CUDA graph
        # capture, then rebinds the committed prefix. Keeping the capacity
        # views lets calls outside a captured graph reuse the kernels compiled
        # at warmup instead of compiling again mid-serving. Only committed
        # blocks are ever indexed, and the uncommitted tail keeps its reserved
        # addresses.
        if self.gdp_replay_capacity_views is None:
            self.gdp_replay_capacity_views = self.kv_cache
        elif is_prefix_view(self.gdp_replay_capacity_views, self.kv_cache):
            self.kv_cache = self.gdp_replay_capacity_views

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None and self.cache_config is not None
        calculator = (
            gdp_replay_state_dtype if self.use_gdp_replayssm else gdp_state_dtype
        )
        return calculator(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        # The helper shards internally, so feed it the unsharded counts.
        calculator = (
            gdp_replay_state_shape if self.use_gdp_replayssm else gdp_state_shape
        )
        return calculator(
            get_tensor_model_parallel_world_size(),
            self.config.gdp_num_groups,
            self.config.gdp_num_heads,
            self.key_head_dim,
            self.head_dim,
            self.conv_kernel_size,
            self.num_householder,
            self.num_spec,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_projected(*self.project(hidden_states))

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        """The input projection, which precedes the GDP op."""
        projected, _ = self.in_proj(hidden_states)
        return (projected,)

    def forward_projected(self, projected: torch.Tensor) -> torch.Tensor:
        # Only the conv and the recurrence need the mamba metadata, so only they
        # sit in the opaque op; inductor fuses the projections' glue and the norm.
        num_tokens = projected.size(0)
        gate, mixed_vkq, b, a = projected.split(
            [
                self.value_dim,
                self.conv_dim,
                self.num_householder * self.num_heads,
                self.num_heads,
            ],
            dim=-1,
        )
        # Zeros: the op leaves padded rows, and all rows of profile runs, unwritten.
        # The replay path zeroes the rows it leaves unwritten itself, which in a
        # captured decode step is none.
        core_out = (torch.empty if self.use_gdp_replayssm else torch.zeros)(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=projected.dtype,
            device=projected.device,
        )
        # The GDP op splits the graph and `gate` crosses it into the next piece; a
        # copy gives it the stride that piece was compiled for, since inductor may
        # pad the projection's rows when their width is not a multiple of 8.
        gate = gate.clone()
        if self.use_gdp_replayssm:
            # The replay op reads the decay logits from the projection itself.
            if self.g_clamp_value is not None:
                a.copy_(self.clamp_decay_input(a))
            torch.ops.vllm.nemotron_shortcut_gdp(
                projected, None, None, core_out, self.prefix
            )
        else:
            if self.g_clamp_value is not None:
                a = self.clamp_decay_input(a)
            # `clone`, not `contiguous`: a single token leaves b/a as one-row views
            # into the projection, which `contiguous` keeps as-is, and the GDP
            # kernels need them 16-byte aligned. Copying here lets inductor fuse
            # the copies.
            torch.ops.vllm.nemotron_shortcut_gdp(
                mixed_vkq, b.clone(), a.clone(), core_out, self.prefix
            )
        core_out = self.norm(core_out.reshape(num_tokens, self.value_dim), gate)
        output, _ = self.out_proj(core_out)
        return output

    def clamp_decay_input(self, a: torch.Tensor) -> torch.Tensor:
        """Apply `g.clamp_min(-gdp_g_clamp_value)` through the decay logits `a`.

        Pretraining lower-bounds the GDP log decay `g` for numerical stability at
        scale. Every GDP kernel (prefill, decode, spec-verify) computes
        `g = -exp(A_log) * softplus(a + dt_bias)` from `a` itself, so the clamp is
        applied to `a` and the kernels stay unchanged. `g` only falls as `a`
        rises, so with `c = gdp_g_clamp_value`, `g >= -c` is exactly the per-head
        bound `a <= softplus^-1(c * exp(-A_log)) - dt_bias
        = log(exp(c * exp(-A_log)) - 1) - dt_bias`, up to rounding the bound to
        the dtype of `a`. Example: c = 5, A_log = 0, dt_bias = 0 gives
        `a <= 4.993` and `g = -softplus(4.993) = -5`.
        """
        assert self.g_clamp_value is not None
        softplus_bound = torch.expm1(self.g_clamp_value * torch.exp(-self.A_log))
        upper = torch.log(softplus_bound.clamp_min(torch.finfo(torch.float32).tiny))
        return torch.minimum(a.float(), upper - self.dt_bias).to(a.dtype)

    def _gdn_metadata(self) -> GDNAttentionMetadata | None:
        """This layer's metadata; None in profile and warmup runs, which carry
        no mamba metadata."""
        context_metadata = get_forward_context().attn_metadata
        metadata = (
            context_metadata.get(self.prefix)
            if isinstance(context_metadata, dict)
            else context_metadata
        )
        return metadata if isinstance(metadata, GDNAttentionMetadata) else None

    def _forward_replay(self, projected: torch.Tensor, core_out: torch.Tensor) -> None:
        """GDP with FlashInfer's `gdp_conv_replay` for decode and spec-verify rows.

        Decode rows take the op on whole in_proj rows (it slices the conv
        channels and gate logits itself); prefill rows take the regular prefill
        and reset their trackers. A row with replay history never reaches the
        prefill path: GDP replay requires spec decode, so every decoding row is
        a spec row, and the metadata builder sends every prefilling row to
        prefill. Plain decode rows appear only in warmup batches.
        """
        md = self._gdn_metadata()
        if md is None:
            core_out.zero_()
            return
        assert self.gate_params is not None and self.gdp_replay_trackers is not None
        commits = self.gdp_replay_trackers.claim(md)
        if (
            self.A_log.data_ptr() != self.gate_params[0].data_ptr()
            or self.dt_bias.data_ptr() != self.gate_params[1].data_ptr()
        ):
            raise RuntimeError(
                f"{self.prefix}: A_log / dt_bias no longer alias the replay gate params"
            )
        # Rows past the step's tokens pad a captured graph; nothing below writes
        # them.
        core_out[md.num_actual_tokens :].zero_()
        if md.spec_sequence_masks is not None:
            self._forward_replay_spec(md, projected, core_out, commits)
            return
        projected = projected[: md.num_actual_tokens]
        core_out = core_out[: md.num_actual_tokens]

        num_decode_tokens = md.num_decode_tokens
        if md.num_decodes > 0:
            # One token per row. With spec decode on, decoding rows are spec
            # rows, so only warmup batches get here; still commit what each
            # row's previous step accepted, not an assumed single token.
            assert md.non_spec_state_indices_tensor is not None
            batch = md.num_decodes
            device = projected.device
            num_accepted = (
                md.num_accepted_tokens[:batch]
                if md.num_accepted_tokens is not None
                else torch.ones(batch, dtype=torch.int32, device=device)
            )
            step = self._replay_step(
                md,
                commits,
                md.non_spec_state_indices_tensor[:batch],
                torch.arange(batch + 1, dtype=torch.int32, device=device),
                num_accepted,
                tokens=1,
                num_rows=num_decode_tokens,
            )
            self._replay(
                projected[:num_decode_tokens], step, core_out[:num_decode_tokens]
            )
        if md.num_prefills > 0:
            if md.num_decodes > 0:
                # The conv's launch metadata covers the decode rows too, so let
                # it derive its own from the prefill rows.
                assert md.prefill_query_start_loc is not None
                assert md.prefill_state_indices is not None
                loc = md.prefill_query_start_loc
                state_indices = md.prefill_state_indices
                has_initial_state = md.prefill_has_initial_state
                conv_metadata = None
            else:
                assert md.non_spec_query_start_loc is not None
                assert md.non_spec_state_indices_tensor is not None
                loc = md.non_spec_query_start_loc
                state_indices = md.non_spec_state_indices_tensor
                has_initial_state = md.has_initial_state
                conv_metadata = md
            self._replay_prefill(
                projected[num_decode_tokens:],
                loc,
                state_indices,
                has_initial_state,
                conv_metadata,
                commits,
                out=core_out[num_decode_tokens:],
            )

    def _forward_replay_spec(
        self,
        md: GDNAttentionMetadata,
        projected: torch.Tensor,
        core_out: torch.Tensor,
        commits: bool,
    ) -> None:
        """A spec-decode step: the replay op on the spec rows, the regular
        prefill on the others (with spec decode on, every non-spec row is a
        prefill).

        The batch reorder puts decoding requests first, so the spec rows are
        usually the step's leading tokens and the prefill tokens follow. Both
        parts then read their rows and write their outputs in place. A short
        prefill chunk can sort among the decodes; such a step gathers and
        scatters its rows."""
        num_spec_tokens = md.num_spec_decode_tokens
        only_spec = md.num_prefills == 0 and md.num_decodes == 0
        if only_spec or _spec_rows_lead(md):
            # The prefill runs after the replay, so it overwrites the +0 the op
            # stores for padding requests on rows that are the prefill's.
            self._replay_spec(md, projected, num_spec_tokens, core_out, commits)
            if not only_spec:
                rows = slice(num_spec_tokens, md.num_actual_tokens)
                self._spec_step_prefill(md, projected[rows], commits, core_out[rows])
            return
        spec_idx, prefill_idx = md.spec_token_indx, md.non_spec_token_indx
        assert spec_idx is not None and prefill_idx is not None
        spec_rows = projected.index_select(0, spec_idx)
        spec_out = self._replay_spec(md, spec_rows, num_spec_tokens, None, commits)
        core_out.index_copy_(0, spec_idx, spec_out)
        prefill_rows = projected.index_select(0, prefill_idx)
        prefill_out = self._spec_step_prefill(md, prefill_rows, commits)
        core_out.index_copy_(0, prefill_idx, prefill_out)

    def _spec_step_prefill(
        self,
        md: GDNAttentionMetadata,
        rows: torch.Tensor,
        commits: bool,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The prefill rows of a spec-decode step."""
        assert md.non_spec_query_start_loc is not None
        assert md.non_spec_state_indices_tensor is not None
        return self._replay_prefill(
            rows,
            md.non_spec_query_start_loc,
            md.non_spec_state_indices_tensor,
            md.has_initial_state,
            conv_metadata=md,
            commits=commits,
            out=out,
        )

    def _replay_spec(
        self,
        md: GDNAttentionMetadata,
        rows: torch.Tensor,
        num_rows: int,
        out: torch.Tensor | None,
        commits: bool,
    ) -> torch.Tensor:
        """Verify windows of up to 1 + num_spec tokens per request: the first
        `num_rows` of `rows`. The op wants them rectangular, so ragged windows
        are padded and the padding is dropped from the output. With `out`, full
        windows write their output there (rows of padding requests get +0);
        `rows` and `out` past `num_rows` may serve as the padding requests'
        rows, which the op does not read."""
        assert md.spec_state_indices_tensor is not None
        assert md.spec_query_start_loc is not None
        assert md.num_accepted_tokens is not None
        batch = md.num_spec_decodes
        step = self._replay_step(
            md,
            commits,
            md.spec_state_indices_tensor[:batch, 0],
            md.spec_query_start_loc,
            md.num_accepted_tokens[:batch],
            tokens=self.num_spec + 1,
            num_rows=num_rows,
        )
        if step.dst is None:
            return self._replay(rows, step, out)
        rows = _pad_spec_window(rows[:num_rows], step.dst, batch * step.tokens)
        result = self._replay(rows, step).index_select(0, step.dst)
        if out is not None:
            out[:num_rows].copy_(result)
        return result

    def _replay_step(
        self,
        md: GDNAttentionMetadata,
        commits: bool,
        slots: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted: torch.Tensor,
        tokens: int,
        num_rows: int,
    ) -> GdpReplayStep:
        """This step's op inputs: the group's first GDP layer commits the
        previous step and plans them; the other layers reuse the plan."""
        trackers = self.gdp_replay_trackers
        assert trackers is not None
        if not commits:
            return trackers.step_of(md)
        batch = slots.size(0)
        batches = self._replay_batches(tokens)
        i = bisect.bisect_left(batches, batch)
        padded = batches[i] if i < len(batches) else batch
        return trackers.commit_step(
            md, slots, query_start_loc, num_accepted, tokens, padded, num_rows
        )

    def _replay_batches(self, tokens: int) -> list[int]:
        """The batch sizes warmup compiled the op for at `tokens` tokens per
        request, ascending: the request counts of the FULL decode graphs. An
        eager step pads up to one, so the op never compiles mid-serving."""
        batches = self._replay_batch_cache.get(tokens)
        if batches is None:
            sizes = set(self.replay_capture_sizes)
            if self.replay_decode_requests:
                # Each decode graph runs n requests as n * tokens tokens; the
                # other captured token counts are prefill graphs, which run the
                # op outside the graph.
                found = {n for n in self.replay_decode_requests if n * tokens in sizes}
            else:
                found = {size // tokens for size in sizes if size % tokens == 0}
            batches = self._replay_batch_cache[tokens] = sorted(found)
        return batches

    def _replay(
        self,
        rows: torch.Tensor,
        step: GdpReplayStep,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run `gdp_conv_replay` on whole in_proj rows, `step.tokens` per
        request, with the step's planned inputs; returns the live requests'
        output rows.

        The padding requests take the rows of `rows` past the live ones, which
        the op does not read, or zero rows if `rows` ends first. When `out` has
        room for every request's rows, the op writes there (+0 on padding
        requests); otherwise the live rows are copied into it."""
        from flashinfer.gdp_conv_replay_decode import gdp_conv_replay

        trackers = self.gdp_replay_trackers
        assert trackers is not None and self.gate_params is not None
        live = step.batch * step.tokens
        size = step.padded * step.tokens
        if rows.size(0) < size:
            rows = torch.cat([rows[:live], rows.new_zeros(size - live, rows.size(1))])
        rows = rows[:size]
        op_out = None
        if out is not None and out.size(0) >= size:
            op_out = out[:size].view(
                step.padded, step.tokens, self.num_heads, self.head_dim
            )
        conv_start = self.value_dim
        gates_start = conv_start + self.conv_dim
        conv_state, state, k_ring, u_ring, g_ring = self.kv_cache
        result = gdp_conv_replay(
            rows[:, conv_start:gates_start],
            rows[:, gates_start:],
            self.conv1d.weight.view(self.conv_dim, self.conv_kernel_size),
            conv_state,
            step.slots,
            self.gate_params,
            state,
            k_ring,
            u_ring,
            g_ring,
            step.slots,
            step.base,
            step.hist_len,
            step.valid_len,
            trackers.seed,
            scale=self.key_head_dim**-0.5,
            # The commit rule in GdpReplayTrackers assumes this threshold.
            flush_min=FLUSH_MIN,
            # The op's tuned default, within about 1 FP16 ulp of exact SiLU.
            silu="tanh",
            ring_bank=step.ring_bank,
            null_block_id=NULL_BLOCK_ID,
            # The norm and out_proj read every row, padding included.
            zero_inactive_out=True,
            out=op_out,
        )
        result = result.view(-1, self.num_heads, self.head_dim)[:live]
        if out is not None and op_out is None:
            out[:live].copy_(result)
        return result

    def _replay_prefill(
        self,
        rows: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        conv_metadata: GDNAttentionMetadata | None,
        commits: bool,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The regular prefill on whole in_proj rows, into `out` if given;
        then, with `commits`, zero the trackers of the prefilled blocks, whose
        checkpoint is now the post-prompt state and whose conv rows 0..2 hold
        the last three prompt inputs."""
        _, mixed_vkq, b, a = rows.split(
            [
                self.value_dim,
                self.conv_dim,
                self.num_householder * self.num_heads,
                self.num_heads,
            ],
            dim=-1,
        )
        assert has_initial_state is not None
        out = self._prefill(
            mixed_vkq,
            b,
            a,
            self.conv1d.weight.view(self.conv_dim, self.conv_kernel_size),
            # The conv kernel takes (dim, rows) views; replay keeps rows first.
            self.kv_cache[0].transpose(-1, -2),
            self.kv_cache[1],
            query_start_loc,
            state_indices,
            has_initial_state,
            conv_metadata,
            out,
        )
        if commits:
            assert self.gdp_replay_trackers is not None
            self.gdp_replay_trackers.reset(state_indices)
        return out

    def _prefill(
        self,
        mixed_vkq: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        conv_metadata: GDNAttentionMetadata | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Conv and chunked GDP over prefill rows; writes each request's last
        three conv inputs and its post-prompt state."""
        mixed_vkq = causal_conv1d_fn(
            mixed_vkq.transpose(0, 1),
            conv_weights,
            None,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=state_indices,
            query_start_loc=query_start_loc,
            metadata=conv_metadata,
        ).transpose(0, 1)
        value, key, query = self._split_heads(mixed_vkq)
        num_tokens = query.size(0)
        decay = -self.A_log.float().exp() * nn.functional.softplus(
            a.float() + self.dt_bias.float()
        )
        beta = b.reshape(num_tokens, self.num_householder, self.num_heads).sigmoid()
        # The kernel accumulates in fp32; a replay checkpoint is stored narrower.
        initial_state = ssm_state[state_indices].float()
        initial_state[~has_initial_state] = 0
        core_out, last_state = gdp_prefill(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
            query_start_loc,
            out,
        )
        ssm_state[state_indices] = last_state.to(ssm_state.dtype)
        return core_out

    def _forward(
        self,
        mixed_vkq: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_out: torch.Tensor,
    ) -> None:
        attn_metadata = self._gdn_metadata()
        if attn_metadata is None:
            return

        num_actual_tokens = attn_metadata.num_actual_tokens
        spec_masks = attn_metadata.spec_sequence_masks
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]

        mixed_vkq = mixed_vkq[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]
        core_out = core_out[:num_actual_tokens]

        # Speculative rows are interleaved with ordinary ones, so peel them out
        # and run the two through their own conv and recurrence.
        only_spec = (
            spec_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        parts: list[
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
        if spec_masks is None:
            parts = [(None, mixed_vkq, b, a)]
        elif only_spec:
            parts = [(attn_metadata.spec_token_indx, mixed_vkq, b, a)]
        else:
            spec_idx = attn_metadata.spec_token_indx
            non_idx = attn_metadata.non_spec_token_indx
            assert spec_idx is not None and non_idx is not None
            spec_vkq, spec_b, spec_a = (
                t.index_select(0, spec_idx) for t in (mixed_vkq, b, a)
            )
            non_vkq, non_b, non_a = (
                t.index_select(0, non_idx) for t in (mixed_vkq, b, a)
            )
            parts = [
                (spec_idx, spec_vkq, spec_b, spec_a),
                (None, non_vkq, non_b, non_a),
            ]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        results: list[tuple[torch.Tensor | None, torch.Tensor]] = []
        for token_idx, vkq, b_part, a_part in parts:
            is_spec = spec_masks is not None and (only_spec or token_idx is not None)
            if is_spec:
                core = self._spec_branch(
                    attn_metadata,
                    vkq,
                    b_part,
                    a_part,
                    conv_weights,
                    conv_state,
                    ssm_state,
                    out=core_out if only_spec else None,
                )
            else:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    continue
                core = self._non_spec_branch(
                    attn_metadata,
                    vkq,
                    b_part,
                    a_part,
                    conv_weights,
                    conv_state,
                    ssm_state,
                    out=core_out if spec_masks is None else None,
                )
            results.append((token_idx if not only_spec else None, core))

        if not results or spec_masks is None:
            return
        if only_spec:
            if results[0][1].data_ptr() != core_out.data_ptr():  # ragged windows
                core_out.copy_(results[0][1])
            return
        spec_idx = attn_metadata.spec_token_indx
        non_idx = attn_metadata.non_spec_token_indx
        assert spec_idx is not None and non_idx is not None
        for idx, part in zip((spec_idx, non_idx), (r[1] for r in results)):
            core_out.index_copy_(0, idx, part)

    def _split_heads(self, mixed_vkq: torch.Tensor, l2norm_qk: bool = True):
        """Split the conv output into value, key and query, L2-normalizing q/k.

        With `l2norm_qk=False` the consumer is the MTP kernel, which normalizes
        q/k itself and reads the strided views as they are.
        """
        num_tokens = mixed_vkq.size(0)
        value, key, query = mixed_vkq.split(
            [
                self.num_householder * self.value_dim,
                self.num_householder * self.key_dim,
                self.key_dim,
            ],
            dim=-1,
        )
        value = value.reshape(
            num_tokens, self.num_householder, self.num_heads, self.head_dim
        )
        key = key.reshape(
            num_tokens, self.num_householder, self.num_groups, self.key_head_dim
        )
        query = query.reshape(num_tokens, self.num_groups, self.key_head_dim)
        if not l2norm_qk:
            return value, key, query
        # The prefill kernel needs contiguous inputs; q/k are normalized here.
        return (
            value.contiguous(),
            l2norm_fwd(key.contiguous()),
            l2norm_fwd(query.contiguous()),
        )

    def _spec_branch(
        self,
        attn_metadata: GDNAttentionMetadata,
        mixed_vkq: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Verify a drafted window; the MTP kernel takes it as T tokens per row."""
        from flashinfer.cake_gdp import cake_gated_delta_product_mtp

        spec_indices = attn_metadata.spec_state_indices_tensor
        num_accepted = attn_metadata.num_accepted_tokens
        query_start_loc = attn_metadata.spec_query_start_loc
        assert spec_indices is not None and num_accepted is not None
        assert query_start_loc is not None

        batch = attn_metadata.num_spec_decodes
        width = spec_indices.size(-1)
        mixed_vkq = causal_conv1d_update(
            mixed_vkq,
            conv_state,
            conv_weights,
            None,
            self.activation,
            conv_state_indices=spec_indices[:, 0][:batch],
            num_accepted_tokens=num_accepted,
            query_start_loc=query_start_loc,
            max_query_len=width,
            validate_data=False,
        )
        # These depend only on the step's metadata, so the first GDP layer builds
        # them and the layers sharing the metadata reuse them. A graph capture
        # builds them again, so that its replays do too.
        cache_key = (mixed_vkq.size(0), torch.cuda.is_current_stream_capturing())
        cached = getattr(attn_metadata, "_gdp_spec_slots", None)
        if cached is not None and cached[0] == cache_key:
            dst, slots, read = cached[1]
        else:
            # A proposer may draft fewer than num_spec tokens, leaving the
            # windows ragged, while the MTP kernel wants a rectangular
            # [batch, width] block.
            dst = None
            if mixed_vkq.size(0) != batch * width:
                starts = query_start_loc[:batch].long()
                lens = query_start_loc[1 : batch + 1].long() - starts
                rows = torch.repeat_interleave(
                    torch.arange(batch, device=lens.device), lens
                )
                offsets = torch.arange(
                    mixed_vkq.size(0), device=lens.device
                ) - torch.repeat_interleave(starts, lens)
                dst = rows * width + offsets

            # Resume from the snapshot taken after the last accepted token; that
            # is what makes a rejected draft roll back rather than corrupt the
            # state. vLLM marks unallocated slots with NULL_BLOCK_ID (0); the
            # kernel skips on -1, so without this it writes state into slot 0,
            # which belongs to another request.
            slots = (
                spec_indices[:batch]
                .to(torch.int32)
                .masked_fill(spec_indices[:batch] == NULL_BLOCK_ID, -1)
            )
            # Gather before masking: num_accepted counts the *previous* window,
            # which may be longer than this one, and a masked slot would read as
            # -1 and silently restart the recurrence from a zero state.
            read = slots.gather(1, (num_accepted[:batch].long() - 1).unsqueeze(1))
            if dst is not None:
                # Padded steps must neither snapshot nor write the end state; -1
                # also lands in slots[:, -1] for short rows, skipping that write.
                padded = torch.ones(batch * width, dtype=torch.bool, device=dst.device)
                padded[dst] = False
                slots = slots.masked_fill(padded.view(batch, width), -1)
            spec_slots = (cache_key, (dst, slots, read))
            attn_metadata._gdp_spec_slots = spec_slots  # type: ignore[attr-defined]
        if dst is not None:
            mixed_vkq, b, a = (
                _pad_spec_window(t, dst, batch * width) for t in (mixed_vkq, b, a)
            )

        value, key, query = self._split_heads(mixed_vkq, l2norm_qk=False)
        core_out, _ = cake_gated_delta_product_mtp(
            q=query.reshape(batch, width, self.num_groups, self.key_head_dim),
            k=key.reshape(
                batch, width, self.num_householder, self.num_groups, self.key_head_dim
            ),
            v=value.reshape(
                batch, width, self.num_householder, self.num_heads, self.head_dim
            ),
            initial_state=ssm_state,
            initial_state_indices=read.squeeze(1).contiguous(),
            A_log=self.A_log.float(),
            a=a.reshape(batch, width, self.num_heads),
            dt_bias=self.dt_bias.float(),
            b=b.reshape(batch, width, self.num_householder, self.num_heads),
            ssm_state_indices=slots,
            disable_state_update=False,
            use_qk_l2norm=True,
            output=None
            if out is None or dst is not None
            else out.view(batch, width, self.num_heads, self.head_dim),
        )
        core_out = core_out.reshape(batch * width, self.num_heads, self.head_dim)
        return core_out if dst is None else core_out.index_select(0, dst)

    def _non_spec_branch(
        self,
        attn_metadata: GDNAttentionMetadata,
        mixed_vkq: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from flashinfer.cake_gdp import cake_gated_delta_product_mtp

        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None and query_start_loc is not None

        if attn_metadata.num_prefills > 0:
            assert attn_metadata.has_initial_state is not None
            return self._prefill(
                mixed_vkq,
                b,
                a,
                conv_weights,
                conv_state,
                ssm_state,
                query_start_loc,
                state_indices,
                attn_metadata.has_initial_state,
                attn_metadata,
                out,
            )

        mixed_vkq = causal_conv1d_update(
            mixed_vkq,
            conv_state,
            conv_weights,
            None,
            self.activation,
            conv_state_indices=state_indices[: attn_metadata.num_decodes],
            validate_data=True,
        )
        # The decode kernel L2-normalizes q/k itself (same rsqrt(sum + 1e-6) as
        # l2norm_fwd, in fp32), saving two kernels per layer and step.
        value, key, query = self._split_heads(mixed_vkq, l2norm_qk=False)
        num_tokens = query.size(0)

        # Decode applies the gating itself, so it takes the raw logits in the
        # model dtype -- unlike prefill, which consumes fp32 gates directly.
        num_decodes = attn_metadata.num_decodes
        decode_indices = state_indices[:num_decodes].to(torch.int32)
        core_out, _ = cake_gated_delta_product_mtp(
            q=query.reshape(num_decodes, 1, self.num_groups, self.key_head_dim),
            k=key.reshape(
                num_decodes, 1, self.num_householder, self.num_groups, self.key_head_dim
            ),
            v=value.reshape(
                num_decodes, 1, self.num_householder, self.num_heads, self.head_dim
            ),
            initial_state=ssm_state,
            initial_state_indices=decode_indices,
            A_log=self.A_log.float(),
            a=a.reshape(num_decodes, 1, self.num_heads),
            dt_bias=self.dt_bias.float(),
            b=b.reshape(num_decodes, 1, self.num_householder, self.num_heads),
            ssm_state_indices=decode_indices.unsqueeze(1),
            disable_state_update=False,
            use_qk_l2norm=True,  # q/k are not normalized above for decode
            output=None
            if out is None
            else out.view(num_decodes, 1, self.num_heads, self.head_dim),
        )
        return core_out.reshape(num_tokens, self.num_heads, self.head_dim)


def nemotron_shortcut_gdp(
    x: torch.Tensor,
    b: torch.Tensor | None,
    a: torch.Tensor | None,
    core_out: torch.Tensor,
    layer_name: str,
) -> None:
    """The GDP conv and recurrence. `x` is the conv input (`mixed_vkq`) with
    its `b` and `a`, or, under `--use-replayssm`, the whole in_proj output
    with no `b` and `a`: the replay op slices them itself."""
    self = get_forward_context().no_compile_layers[layer_name]
    if b is None or a is None:
        self._forward_replay(x, core_out)
    else:
        self._forward(x, b, a, core_out)


direct_register_custom_op(
    op_name="nemotron_shortcut_gdp",
    op_func=nemotron_shortcut_gdp,
    mutates_args=["core_out"],
)
