# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated DeltaProduct (GDP) linear-attention mixer for Nemotron Shortcut.

GDP is a gated delta rule that takes `gdp_num_householder` delta steps per
token. It reuses the GDN attention backend for its metadata and FlashInfer's
GDP kernels: `gdp_prefill.chunk_gated_delta_product` for prefill and
`cake_gdp.cake_gated_delta_product_mtp` for decode and speculative verify.
"""

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
from vllm.utils.torch_utils import direct_register_custom_op, get_kv_cache_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


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
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
        )
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

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None and self.cache_config is not None
        return gdp_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        # The helper shards internally, so feed it the unsharded counts.
        return gdp_state_shape(
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
        if self.g_clamp_value is not None:
            a = self.clamp_decay_input(a)
        # Zeros: the op leaves padded rows, and all rows of profile runs, unwritten.
        core_out = torch.zeros(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=projected.dtype,
            device=projected.device,
        )
        # `clone`, not `contiguous`: a single token leaves b/a as one-row views into
        # the projection, which `contiguous` keeps as-is, and the GDP kernels need
        # them 16-byte aligned. Copying here lets inductor fuse the copies.
        # The GDP op splits the graph and `gate` crosses it into the next piece; a
        # copy gives it the stride that piece was compiled for, since inductor may
        # pad the projection's rows when their width is not a multiple of 8.
        gate = gate.clone()
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

    def _forward(
        self,
        mixed_vkq: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_out: torch.Tensor,
    ) -> None:
        context_metadata = get_forward_context().attn_metadata
        metadata = (
            context_metadata.get(self.prefix)
            if isinstance(context_metadata, dict)
            else context_metadata
        )
        if not isinstance(metadata, GDNAttentionMetadata):
            # Profile and warmup runs carry no mamba metadata.
            return
        attn_metadata = metadata

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
        if spec_masks is None:
            parts = [(None, mixed_vkq, b, a)]
        elif only_spec:
            parts = [(attn_metadata.spec_token_indx, mixed_vkq, b, a)]
        else:
            spec_idx = attn_metadata.spec_token_indx
            non_idx = attn_metadata.non_spec_token_indx
            assert spec_idx is not None and non_idx is not None
            parts = [
                (spec_idx, *(t.index_select(0, spec_idx) for t in (mixed_vkq, b, a))),
                (None, *(t.index_select(0, non_idx) for t in (mixed_vkq, b, a))),
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
            mixed_vkq = causal_conv1d_fn(
                mixed_vkq.transpose(0, 1),
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=attn_metadata.has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        else:
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
        value, key, query = self._split_heads(
            mixed_vkq, l2norm_qk=attn_metadata.num_prefills > 0
        )
        num_tokens = query.size(0)

        if attn_metadata.num_prefills > 0:
            decay = -self.A_log.float().exp() * nn.functional.softplus(
                a.float() + self.dt_bias.float()
            )
            beta = b.reshape(num_tokens, self.num_householder, self.num_heads).sigmoid()
            has_initial_state = attn_metadata.has_initial_state
            assert has_initial_state is not None
            initial_state = ssm_state[state_indices].contiguous()
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
    mixed_vkq: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_out: torch.Tensor,
    layer_name: str,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward(mixed_vkq, b, a, core_out)


direct_register_custom_op(
    op_name="nemotron_shortcut_gdp",
    op_func=nemotron_shortcut_gdp,
    mutates_args=["core_out"],
)
