# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN attention metadata for RecoverSSM speculative decode (Qwen-style Gated DeltaNet).

Wraps the generic GDN builder: every active decode row runs through the spec path (a
draft-less step is a T=1 window), the spec state indices keep only the checkpoint
column, num_accepted_tokens is always 1 (the checkpoint and the compacted conv window
already hold the accepted state), and the metadata carries what the post-sampling commit
needs. Mirrors the Kimi-K3 KDA builder's RecoverSSM handling."""

from dataclasses import dataclass, field, fields
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    recoverssm_request_indices,
    recoverssm_spec_rows,
)

logger = init_logger(__name__)


@dataclass
class GDNRecoverSSMCommitMetadata:
    state_indices: torch.Tensor  # [num_spec_decodes, 1]
    query_start_loc: torch.Tensor  # [num_spec_decodes + 1]
    request_indices: (
        torch.Tensor | None
    )  # batch rows of the spec requests (None: rows 0..n-1)
    block_table: torch.Tensor | None  # align mode
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclass
class GDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: GDNRecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        c = self.recoverssm_commit
        if c is None:
            return None
        n = self.num_spec_decodes
        self.recoverssm_context.commit(
            num_accepted_tokens,
            c.state_indices[:n, 0],
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


class GDNRecoverSSMMetadataBuilder(GDNAttentionMetadataBuilder):
    # The verify/commit metadata is not staged for FULL graph replay yet; declaring
    # NEVER makes the runner fall back to PIECEWISE (attention compiled piecewise).
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The builder may be created before the runner resolves the CUDA graph mode.
        self.use_full_cuda_graph = False
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ones = torch.ones(max_reqs, dtype=torch.int32, device=device)
        self._context: Any = None
        self._logged = False

    def _get_context(self):
        if self._context is None:
            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
                GDNRecoverSSMCommitContext,
            )

            fc = self.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in self.layer_names]
            self._context = GDNRecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            logger.info(
                "GDN RecoverSSM commit context: %d layers, spec_query_len %d",
                len(layers),
                1 + self.num_spec,
            )
        return self._context

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> "GDNRecoverSSMMetadata":
        m = common_attn_metadata
        rows = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("GDN RecoverSSM needs the CPU is_prefilling mask")
            # Positive counts only defeat the base builder's "no drafts -> regular
            # decode" shortcut: RecoverSSM keeps every post-prefill row on the spec path
            # (extended conv window).
            num_decode_draft_tokens_cpu, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                m.num_reqs,
            )
        meta = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        commit = None
        if meta.num_spec_decodes > 0:
            n = meta.num_spec_decodes
            assert meta.spec_state_indices_tensor is not None
            base["spec_state_indices_tensor"] = meta.spec_state_indices_tensor[:, :1]
            base["num_accepted_tokens"] = self._ones[:n]
            if rows is None:
                raise ValueError("GDN RecoverSSM: spec rows without draft counts")
            request_indices = recoverssm_request_indices(
                rows, n, m.query_start_loc.device
            )
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            commit = GDNRecoverSSMCommitMetadata(
                state_indices=base["spec_state_indices_tensor"],
                query_start_loc=meta.spec_query_start_loc,
                request_indices=request_indices,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=m.compute_num_computed_tokens() if align else None,
                block_size=self.kv_cache_spec.block_size if align else None,
            )
            if not self._logged:
                self._logged = True
                logger.info(
                    "GDN RecoverSSM path taken: %d spec rows, align=%s", n, align
                )
        return GDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_context=self._get_context() if commit is not None else None,
        )


class GDNRecoverSSMAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[GDNRecoverSSMMetadataBuilder]:
        return GDNRecoverSSMMetadataBuilder
