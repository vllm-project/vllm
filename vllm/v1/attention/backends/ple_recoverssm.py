# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM protocol for the Qwen4Exp PLE short-conv layer.

When the GDN layers run RecoverSSM, the runner resets the shared num_accepted_tokens to
1 after each commit (align mode). The PLE dilated short conv keeps an extended window in
one state block and reads it at num_accepted - 1, so it must follow the same protocol:
every active decode row takes the spec path, the conv reads at offset 0, and after
sampling the accepted window is compacted to the front (the shared RecoverSSM compaction
kernel; in align mode it also writes the block-boundary window)."""

from dataclasses import dataclass, field, fields
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.recoverssm import (
    compact_conv_state_kernel,
    prepare_commit_plan_kernel,
)
from vllm.model_executor.layers.mamba.recoverssm_utils import recoverssm_require
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    recoverssm_request_indices,
    recoverssm_spec_rows,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)


def _require(cond: bool, msg: str) -> None:
    recoverssm_require(cond, msg, "PLE RecoverSSM")


class _PleConvCommit:
    def __init__(self, conv_states, spec_query_len: int, max_num_reqs: int):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        _require(len(conv_states) > 0, "PLE commit requires at least one layer")
        if not is_conv_state_dim_first():
            conv_states = [
                s.transpose(-1, -2) for s in conv_states
            ]  # -> [blocks, C, W]
        self.conv_states = list(conv_states)
        ref = self.conv_states[0]
        _require(ref.ndim == 3, "PLE conv state must be [blocks, dim, window]")
        dev = ref.device
        for s in self.conv_states:
            _require(
                s.shape == ref.shape
                and s.dtype == ref.dtype
                and s.stride()[1:] == ref.stride()[1:]
                and s.device == dev,
                "PLE layers need matching conv states",
            )
        self.conv_dim, conv_len = ref.shape[1], ref.shape[2]
        self.hist = conv_len - spec_query_len + 1
        _require(self.hist > 0, "PLE conv state is shorter than its window")
        self.spec_query_len = spec_query_len
        t = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
        self.base = t([s.data_ptr() for s in self.conv_states])
        self.bstride = t([s.stride(0) for s in self.conv_states])
        self.dstride = t([s.stride(1) for s in self.conv_states])
        self.tstride = t([s.stride(2) for s in self.conv_states])
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len = (
            z(),
            z(),
            z(),
            z(),
        )

    def commit(
        self,
        num_accepted_tokens,
        state_indices,
        query_start_loc,
        request_indices=None,
        block_table=None,
        num_computed_tokens=None,
        mamba_block_size=None,
    ):
        batch = state_indices.shape[0]
        if batch == 0:
            return
        _require(state_indices.ndim == 1, "state indices must be one-dimensional")
        _require(
            batch <= self.commit_lens.shape[0],
            "PLE commit batch exceeds its plan capacity",
        )
        _require(
            query_start_loc.ndim == 1 and query_start_loc.shape[0] == batch + 1,
            "PLE commit metadata is incompatible",
        )
        _require(
            request_indices is None or request_indices.shape[0] >= batch,
            "PLE request mapping is too short",
        )
        _require(
            num_accepted_tokens.ndim == 1
            and (request_indices is not None or num_accepted_tokens.shape[0] >= batch),
            "PLE accepted-token counts are too short",
        )
        _align = (block_table, num_computed_tokens, mamba_block_size)
        _require(
            all(x is None for x in _align) or all(x is not None for x in _align),
            "PLE align metadata is incomplete",
        )
        _require(
            mamba_block_size is None or mamba_block_size >= self.spec_query_len,
            "PLE align block size must cover one speculative window",
        )
        _require(
            block_table is None or block_table.ndim == 2,
            "PLE block table must be two-dimensional",
        )
        _dev = self.conv_states[0].device
        _require(
            all(
                t_ is None or t_.device == _dev
                for t_ in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                    request_indices,
                    block_table,
                    num_computed_tokens,
                )
            ),
            "PLE commit inputs must be on the same device",
        )
        bt = (0, 0) if block_table is None else block_table.stride()
        prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_idx,
            self.bnd_idx,
            self.bnd_len,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            bt[0],
            bt[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        compact_conv_state_kernel[
            (triton.cdiv(self.conv_dim, 256), batch, len(self.conv_states))
        ](
            self.conv_states[0],
            self.base,
            self.bstride,
            self.dstride,
            self.tstride,
            state_indices,
            self.commit_lens,
            self.final_idx,
            self.bnd_idx,
            self.bnd_len,
            NULL_BLOCK_ID,
            self.conv_dim,
            self.hist,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.hist),
            ALIGN_MODE=block_table is not None,
            num_warps=4,
        )


@dataclass
class PleRecoverSSMCommitMetadata:
    state_indices: torch.Tensor  # [n]
    query_start_loc: torch.Tensor  # [n + 1]
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclass
class PleRecoverSSMMetadata(PleShortConvAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: PleRecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(self, num_accepted_tokens):
        c = self.recoverssm_commit
        if c is None:
            return None
        n = self.num_spec_decodes
        self.recoverssm_context.commit(
            num_accepted_tokens,
            c.state_indices[:n],
            c.query_start_loc[: n + 1],
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            mamba_block_size=c.block_size,
        )
        if c.block_table is None:
            return None
        assert c.block_size is not None and c.num_computed_tokens is not None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=n,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            block_size=c.block_size,
        )


class PleRecoverSSMMetadataBuilder(PleShortConvAttentionMetadataBuilder):
    # Same as the GDN RecoverSSM builder: uniform spec-decode batches replay as FULL
    # graphs on the base builder's staged, null-padded buffers; the commit runs after
    # sampling, outside the graph.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._ones = torch.ones(
            vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device
        )
        self._ctx: _PleConvCommit | None = None
        self._logged = False
        self._capturing = False

    def _get_ctx(self):
        if self._ctx is None:
            fc = self.vllm_config.compilation_config.static_forward_context
            convs = [fc[name].kv_cache[0] for name in self.layer_names]
            ctx = _PleConvCommit(
                convs, 1 + self.num_spec, self.vllm_config.scheduler_config.max_num_seqs
            )
            self._ctx = ctx
            logger.info(
                "PLE RecoverSSM commit context: %d layers, history %d",
                len(convs),
                ctx.hist,
            )
        return self._ctx

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        *,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        **kwargs,
    ):  # type: ignore[override]
        m = common_attn_metadata
        rows = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            if self._capturing and m.is_prefilling is None:
                # CUDA graph capture: dummy decode batches, every row a spec window.
                m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("PLE RecoverSSM needs the CPU is_prefilling mask")
            # the same classification as the GDN builder, so both commits map the same
            # requests
            num_decode_draft_tokens_cpu, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                m.num_reqs,
            )
        meta = super().build(
            common_prefix_len,
            m,
            fast_build,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            **kwargs,
        )
        if (
            not isinstance(meta, PleShortConvAttentionMetadata)
            or meta.num_spec_decodes == 0
        ):
            return meta
        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        n = meta.num_spec_decodes
        # sized like the (possibly graph-padded) state indices
        assert meta.spec_state_indices_tensor is not None
        num_rows = meta.spec_state_indices_tensor.shape[0]
        base["num_accepted_tokens"] = self._ones[:num_rows]
        if rows is None:
            raise ValueError("PLE RecoverSSM: spec rows without draft counts")
        req_idx = recoverssm_request_indices(rows, n, m.query_start_loc.device)
        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        commit = PleRecoverSSMCommitMetadata(
            state_indices=meta.spec_state_indices_tensor,
            query_start_loc=meta.spec_query_start_loc,
            request_indices=req_idx,
            block_table=m.block_table_tensor if align else None,
            num_computed_tokens=m.compute_num_computed_tokens() if align else None,
            block_size=self.kv_cache_spec.block_size if align else None,
        )
        if not self._logged:
            self._logged = True
            logger.info("PLE RecoverSSM path taken: %d spec rows, align=%s", n, align)
        return PleRecoverSSMMetadata(
            **base, recoverssm_commit=commit, recoverssm_context=self._get_ctx()
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False


class PleRecoverSSMAttentionBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PLE_SHORT_CONV_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[PleRecoverSSMMetadataBuilder]:
        return PleRecoverSSMMetadataBuilder
