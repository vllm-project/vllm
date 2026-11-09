# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How the specialized decode installs its deferred RecoverSSM on the stock Kimi K3
objects, without changing their classes:

- the KDA layers (`install_layers`, before the specialized providers are built):
  verify records indexed by record slot (with a scratch row) and a verify that
  flushes the pending replays and writes the records to each request's record slot;
- the model runner state (`install_model_state`): a `DeferredRecoverSSMState`,
  whose bind sets the builders up, with the step's request slots, in place of the
  stock deferred commit;
- the KDA metadata builders (`install_builder`, on first bind): the
  `fp32_deferred` commit context, and each step's `recoverssm_deferred`."""

from types import MethodType
from typing import Any

import torch
from einops import rearrange

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup

from .context import (
    FP32_DEFERRED,
    KDADeferredCommitContext,
    KDARecoverSSMDeferred,
    allocate_records,
)
from .kernels import kda_recoverssm_verify


class DeferredRecoverSSMState(RecoverSSMState):
    """The runner state's RecoverSSM with the specialized decode: its `pending`
    counts are those of the KDA metadata builders `install_builder` sets up, which
    defer the commit their own way instead of the stock one."""

    def bind(
        self,
        attn_groups: list[list[AttentionGroup]],
        request_slots: torch.Tensor | None,
    ) -> None:
        """Hand deferring builders the step's request slots (None: capture or
        dummy batches, whose rows own no request) and, once, the pending counts.
        Only ubatch 0's builders: the hybrid state does not support DBO."""
        for group_list in attn_groups:
            for group in group_list:
                builder = group.get_metadata_builder(0)
                if not isinstance(builder, KimiK3KDAMetadataBuilder):
                    continue
                if not builder.use_recoverssm:
                    continue
                if getattr(builder, "recoverssm_pending", None) is not self.pending:
                    install_builder(builder, self.pending)
                builder.recoverssm_request_slots = request_slots


def install_model_state(state: Any) -> None:
    """Install the deferred RecoverSSM on a `KimiK3ModelState` (no-op without
    RecoverSSM): its `recoverssm` becomes a `DeferredRecoverSSMState`, which the
    base state resets per new request and binds before the metadata build."""
    if state.recoverssm is not None:
        state.recoverssm = DeferredRecoverSSMState(state.max_num_reqs, state.device)


class _DeferredStepCommit:
    """A step's commit context (the KDA metadata's `recoverssm_context`): the
    `fp32_deferred` context's commit with the step's deferred operands."""

    def __init__(
        self, context: KDADeferredCommitContext, deferred: KDARecoverSSMDeferred
    ) -> None:
        self.context, self.deferred = context, deferred

    def commit(self, *args, **kwargs) -> None:
        deferred = self.deferred
        kwargs.update(
            record_rows=deferred.record_rows,
            pending=deferred.pending,
            eager=deferred.eager,
        )
        self.context.commit(*args, **kwargs)  # type: ignore[misc]


def install_builder(builder: KimiK3KDAMetadataBuilder, pending: torch.Tensor) -> None:
    """Set up a KDA metadata builder for the deferred RecoverSSM: the
    `fp32_deferred` commit context for Triton RecoverSSM layers (CuTe ones keep
    their own FP32-state commit), and a build that adds the step's
    `recoverssm_deferred` (None: not deferred) to the metadata."""
    builder.recoverssm_pending = pending
    builder.recoverssm_request_slots = None
    if getattr(builder, "recoverssm_record_rows", None) is not None:
        return  # rebound to new pending counts
    stale = builder.recoverssm_context
    # A stock Triton context built before the first bind (no runner path does
    # today) would pin the stock commit: rebuild it as the deferred one.
    if (
        stale is not None
        and not isinstance(stale, KDADeferredCommitContext)
        and stale.backend == "triton"
    ):
        builder.recoverssm_context = None
    max_num_reqs = builder.vllm_config.scheduler_config.max_num_seqs
    # Each speculative row's record slot (staged, so FULL CUDA graphs read the
    # step's values; -1 past the real rows).
    builder.recoverssm_record_rows = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=builder.device
    )
    base_get_context = builder._get_recoverssm_context
    base_build = builder.build

    def get_context():
        if builder.recoverssm_context is not None:
            return builder.recoverssm_context
        compilation_config = builder.vllm_config.compilation_config
        forward_context = compilation_config.static_forward_context
        layers = [forward_context[name] for name in builder.layer_names]
        if layers[0].recoverssm_backend != "triton":
            return base_get_context()
        context = KDADeferredCommitContext.create(
            layers,
            spec_query_len=1 + builder.vllm_config.num_speculative_tokens,
            max_num_reqs=max_num_reqs,
        )
        builder.recoverssm_context = context
        return context

    def build(
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        metadata = base_build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        deferred = None
        commit = metadata.recoverssm_commit
        if commit is not None and builder.recoverssm_pending is not None:
            context = metadata.recoverssm_context
            assert context is not None
            if context.backend == FP32_DEFERRED:
                assert isinstance(context, KDADeferredCommitContext)
                deferred = _build_deferred(
                    builder,
                    common_attn_metadata,
                    context,
                    metadata.num_spec_decodes,
                    commit.request_indices,
                    num_decode_draft_tokens_cpu,
                    commit.state_indices[:, 0],
                )
                step = _DeferredStepCommit(context, deferred)
                metadata.recoverssm_context = step  # type: ignore[assignment]
        metadata.recoverssm_deferred = deferred  # type: ignore[assignment]
        return metadata

    builder._get_recoverssm_context = get_context  # type: ignore[method-assign]
    builder.build = build  # type: ignore[method-assign]


def _spec_sequence_masks_cpu(
    builder: KimiK3KDAMetadataBuilder, m, num_decode_draft_tokens_cpu: torch.Tensor
) -> torch.Tensor:
    """The speculative rows of a RecoverSSM build with speculative rows (the
    builder's `spec_sequence_masks_cpu`)."""
    query_lens_cpu = m.query_start_loc_cpu.diff()
    masks = num_decode_draft_tokens_cpu >= 0
    masks |= (~m.is_prefilling) & (query_lens_cpu > 0)
    masks &= query_lens_cpu <= builder.num_spec + 1
    return masks


def _build_deferred(
    builder: KimiK3KDAMetadataBuilder,
    m,
    context: KDADeferredCommitContext,
    num_spec_decodes: int,
    spec_request_indices: torch.Tensor | None,
    num_decode_draft_tokens_cpu: torch.Tensor,
    state_indices: torch.Tensor,
) -> KDARecoverSSMDeferred:
    """Stage the speculative rows' record slots; with prefix caching (align),
    mark the rows to commit after sampling: those whose window, up to the next
    step's, may reach a mamba block boundary (the boundary state and the next
    block's checkpoint must hold committed states), from the host's upper
    bound of each sequence length with a margin for rejected drafts."""
    pending = builder.recoverssm_pending
    assert pending is not None
    rows = builder.recoverssm_record_rows
    slots = builder.recoverssm_request_slots
    live = 0
    if slots is not None:
        if spec_request_indices is not None:
            slots = slots[spec_request_indices]
        live = min(num_spec_decodes, slots.shape[0])
        rows[:live].copy_(slots[:live], non_blocking=True)
    rows[live : max(m.num_reqs, num_spec_decodes)].fill_(-1)
    eager = None
    if builder.vllm_config.cache_config.mamba_cache_mode == "align":
        assert isinstance(builder.kv_cache_spec, MambaSpec)
        block = builder.kv_cache_spec.block_size
        margin = 2 * (builder.num_spec + 1)
        upper = m.seq_lens_cpu_upper_bound
        query_lens = m.query_start_loc_cpu.diff()
        masks = _spec_sequence_masks_cpu(builder, m, num_decode_draft_tokens_cpu)
        spec_rows = masks.nonzero(as_tuple=True)[0]
        if upper is None:
            near = torch.ones(num_spec_decodes, dtype=torch.bool)
        else:
            end = upper[spec_rows].to(torch.int64)
            low = end - query_lens[spec_rows].to(torch.int64) - margin
            high = end + margin
            near = (low < 0) | (low.clamp(min=0) // block != high // block)
        if bool(near.any()):
            eager = async_tensor_h2d(near.to(torch.int32), device=state_indices.device)
    return KDARecoverSSMDeferred(
        record_rows=rows,
        pending=pending,
        eager=eager,
        context=context,
        state_indices=state_indices[:num_spec_decodes],
    )


def install_layers(layers) -> None:
    """Before the specialized providers read them: give every Triton RecoverSSM KDA
    layer of `layers` (decoder layers) verify records indexed by record slot and
    the deferred verify (`deferred_recoverssm_verify`). CuTe RecoverSSM layers
    keep their own FP32-state commit and are left as they are."""
    for decoder in layers:
        attn: Any = getattr(decoder, "self_attn", None)
        records = getattr(attn, "recoverssm_records", None)
        if records is None or attn.recoverssm_backend != "triton":
            continue
        if getattr(attn, "deferred_recoverssm", False):
            continue
        rows, heads, window, head_dim = records.correction.shape
        attn.recoverssm_records = allocate_records(
            rows,
            heads,
            window,
            head_dim,
            records.key_gate.dtype,
            records.key_gate.device,
        )
        attn._recoverssm_verify = MethodType(deferred_recoverssm_verify, attn)
        attn.deferred_recoverssm = True


def deferred_recoverssm_verify(
    self,
    m,
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """`KimiK3DeltaAttention._recoverssm_verify` of a Triton RecoverSSM layer
    under the deferred RecoverSSM: a deferred step's verify does not replay the
    last step's accepted tokens, so it folds them into the checkpoints first (every
    layer, once per step), and writes its records to the request slots. Other steps
    run the stock verify."""
    deferred = getattr(m, "recoverssm_deferred", None)
    if deferred is None:
        return type(self)._recoverssm_verify(
            self, m, mixed_qkv, raw_g, raw_beta, conv_weight, query_start_loc, out
        )
    num_tokens = mixed_qkv.shape[0]
    if out is None:
        out = mixed_qkv.new_empty((1, num_tokens, self.local_num_heads, self.head_dim))
    assert m.spec_state_indices_tensor is not None
    assert m.num_accepted_tokens is not None
    state_indices = m.spec_state_indices_tensor[: m.num_spec_decodes, 0]
    records = self.recoverssm_records
    assert records is not None
    conv_state, state = self.kv_cache[:2]
    deferred.flush()
    record_indices = deferred.record_rows[: m.num_spec_decodes]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        self.conv1d.bias,
        activation="silu",
        conv_state_indices=state_indices,
        num_accepted_tokens=m.num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=self.spec_query_len,
        validate_data=False,
        out=torch.empty_like(mixed_qkv),
    )
    q, k, v = (
        rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
        for x in mixed_qkv.split(self.local_projection_size, dim=-1)
    )
    return kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        lower_bound=self.gate_lower_bound,
        checkpoint_state=state,
        records=records,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=self.spec_query_len,
        out=out,
        record_indices=record_indices,
    )
