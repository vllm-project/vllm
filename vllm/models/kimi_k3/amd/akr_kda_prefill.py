# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels KDA prefill (entry `attention/kda_common.kda_prefill`) for
`KimiK3DeltaAttention._forward`.

One call replaces, for the prefill tokens of a layer: the fused qkv causal conv + SiLU,
the gate / beta / q-k L2-norm computation, `chunk_kda_prefill` and the gated RMSNorm
(`self.o_norm`). Route `kda` of `akr_config`, sub-switch `VLLM_KIMI_AKR_KDA_PREFILL`
(default off: unset keeps the stock path, `=1` enables it). Off by default because of an
open bug: with it on, served requests can be corrupted across requests (garbage answers
in GSM8K runs at TP8/DCP8).

Constraints (any failure falls back to the stock path before anything is written):
- The entry serves the call (`kda_prefill_unsupported_reason`: the (heads, head_dim) of
  the recurrent state, bf16 activations, fp32 state, a row stride of the in_proj output
  divisible by 8, the gate / state / index layouts).
- No spec-decode rows in the batch (`spec_sequence_masks is None`): a mixed prefill +
  spec-verify step takes the stock path.
- The hand-over flags of the one-launch path are one persistent buffer per device: two
  KDA prefill layers must never run concurrently on two streams (KDA layers run on the
  main stream only).
Numerics: prefill rounds conv+SiLU and o to bf16 like stock; decode keeps fp32 (about 1
bf16 ulp apart). Mixed steps (decode-first batch): the length-1 decodes take the stock
`causal_conv1d_update` + packed decode recurrence (+ gated norm); the prefill sequences
take this kernel.
"""

import os

import torch

from vllm.logger import init_logger

from . import akr_config

logger = init_logger(__name__)

_ATTR_W = "_akr_kda_conv_w"
_STEP = "_akr_kda_table"


def _env_on() -> bool:
    """VLLM_KIMI_AKR_KDA_PREFILL: off unless set to 1/true/on/yes (open cross-request
    corruption, see above)."""
    v = os.environ.get("VLLM_KIMI_AKR_KDA_PREFILL", "").strip().lower()
    return v in ("1", "true", "on", "yes")


def _impl() -> str | None:
    """VLLM_KIMI_AKR_KDA_PREFILL_IMPL: auto (default: the kernel's uses_mega choice),
    mega or v1 (staged)."""
    v = os.environ.get("VLLM_KIMI_AKR_KDA_PREFILL_IMPL", "auto").strip().lower()
    return v if v in ("mega", "v1") else None


def chunk_counts_of(query_lens_cpu: torch.Tensor, chunk: int = 64) -> list[int]:
    return ((query_lens_cpu + (chunk - 1)) // chunk).tolist()


def attach_chunk_counts(attn_metadata, query_start_loc_cpu: torch.Tensor) -> None:
    """Host chunk counts of the sequences the chunk metadata was built for (metadata
    builder hook)."""
    if attn_metadata.num_prefills <= 0 or attn_metadata.chunk_indices is None:
        return
    lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    if attn_metadata.spec_sequence_masks_cpu is not None:
        lens = lens[~attn_metadata.spec_sequence_masks_cpu]
    elif attn_metadata.num_decodes > 0:
        lens = lens[attn_metadata.num_decodes :]
    attn_metadata.akr_chunk_counts = chunk_counts_of(lens)


def _unsupported(m) -> str | None:
    if not akr_config.on("kda") or not _env_on():
        return "off"
    if m.spec_sequence_masks is not None:
        return "spec"
    if getattr(m, "akr_chunk_counts", None) is None or m.chunk_indices is None:
        return "no chunk counts"
    return None


def _conv_weight(layer) -> torch.Tensor:
    w = getattr(layer, _ATTR_W, None)
    if w is None:
        w = (
            layer.conv1d.weight.detach()
            .view(3, layer.local_projection_size, layer.conv_size)
            .transpose(1, 2)
            .contiguous()
            .float()
        )
        setattr(layer, _ATTR_W, w)
    return w


def _table(m, prefill_state_indices, prefill_has_initial_state, cu, state_cache):
    from amd_kernels.attention.kda_common import kda_prefill_chunk_table

    t = getattr(m, _STEP, None)
    if t is None:
        t = kda_prefill_chunk_table(
            cu,
            m.chunk_indices,
            prefill_state_indices,
            prefill_has_initial_state,
            state_cache=state_cache,
        )
        setattr(m, _STEP, t)
    return t


def try_prefill(
    layer,
    m,
    mixed_qkv_ns,
    g1_ns,
    beta_ns,
    g2,
    conv_state,
    recurrent_state,
    core_attn_out,
    num_actual_tokens,
) -> bool:
    """Run the non-spec part (decode-first decodes + prefills) of a layer. Writes
    `core_attn_out[:, :num_actual_tokens]` (already gated-normed). Returns False,
    touching nothing, when the stock path must run."""
    if _unsupported(m) is not None:
        return False
    try:
        from amd_kernels.attention.kda_common import (
            kda_prefill,
            kda_prefill_unsupported_reason,
        )
    except ImportError:
        return False

    nd = m.num_decodes  # decode-first, only when there is no spec split (checked above)
    nd_tok = m.num_decode_tokens
    cu = m.prefill_query_start_loc
    sidx = m.prefill_state_indices
    has = m.prefill_has_initial_state
    if cu is None or sidx is None or has is None:
        return False
    checkpoint = getattr(m, "checkpoint", None)
    ck = {}
    if checkpoint is not None:
        ck = dict(
            checkpoint_state=recurrent_state,
            checkpoint_offsets=checkpoint.checkpoint_offsets,
            checkpoint_rows=checkpoint.state_indices,
        )
    args = (
        mixed_qkv_ns[nd_tok:num_actual_tokens],
        g1_ns[0, nd_tok:num_actual_tokens],
        beta_ns[0, nd_tok:num_actual_tokens],
        g2[nd_tok:num_actual_tokens],
        layer.A_log,
        layer.dt_bias,
        _conv_weight(layer),
        conv_state,
        recurrent_state,
        sidx,
        has,
        cu,
        m.chunk_indices,
        m.chunk_offsets,
        layer.o_norm.weight,
        layer.o_norm.eps,
        core_attn_out[0, nd_tok:num_actual_tokens],
        layer.gate_lower_bound,
    )
    kw = dict(chunk_counts=m.akr_chunk_counts, impl=_impl(), **ck)
    if akr_config.call_unsupported("kda", kda_prefill_unsupported_reason(*args, **kw)):
        return False

    if nd:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            fused_recurrent_kda_packed_decode,
        )

        idx = m.non_spec_state_indices_tensor[:nd]
        conv_weights = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
        )
        dec_in = mixed_qkv_ns[:nd_tok]
        conv_out = torch.empty_like(dec_in)
        dec_in = causal_conv1d_update(
            dec_in,
            conv_state,
            conv_weights,
            layer.conv1d.bias,
            activation="silu",
            conv_state_indices=idx,
            validate_data=True,
            out=conv_out,
        )
        dec_out, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=dec_in,
            raw_g=g1_ns[:, :nd_tok],
            raw_beta=beta_ns[:, :nd_tok],
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=recurrent_state,
            state_indices=idx,
        )
        core_attn_out[0, :nd_tok] = dec_out[0, :nd_tok]
        core_attn_out[:, :nd_tok] = layer.o_norm(core_attn_out[:, :nd_tok], g2[:nd_tok])

    kda_prefill(*args, table=_table(m, sidx, has, cu, recurrent_state), **kw)
    if checkpoint is not None:
        # conv_state is final (post-conv) exactly as when stock exports after its conv
        # launch.
        layer._checkpoint_exporter.export(
            checkpoint,
            raw_qkv=mixed_qkv_ns[nd_tok:num_actual_tokens],
            conv_state=conv_state,
            cu_seqlens=cu,
        )
    return True
