# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA metadata for RecoverSSM speculative decoding.

RecoverSSM verifies a draft window from a read-only recurrent-state checkpoint
and folds only the accepted tokens into it after sampling, so each request keeps
one state slot instead of one per draft position. GLM-5.3 reuses Kimi-K3's KDA
RecoverSSM kernels; this module adapts the shared GDN metadata to them.
"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KDARecoverSSMAlignMetadata,
    KDARecoverSSMCommitMetadata,
    KimiK3KDAMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
    )


@dataclass(frozen=True)
class KDARecoverSSMLayerView:
    """A GLM-5.3 KDA layer in the parameter layout Kimi-K3's commit expects."""

    kv_cache: Any
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    local_num_heads: int
    head_dim: int
    gate_lower_bound: float | None

    @classmethod
    def of(cls, layer: Any) -> "KDARecoverSSMLayerView":
        return cls(
            kv_cache=layer.kv_cache,
            # GLM-5.3 stores A_log as [1, 1, heads, 1].
            A_log=layer.A_log.view(-1),
            dt_bias=layer.dt_bias,
            local_num_heads=layer.local_num_heads,
            head_dim=layer.head_dim,
            gate_lower_bound=layer.kda_lower_bound,
        )


@dataclass
class Glm5NextKDAMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: KDARecoverSSMCommitMetadata | None = None
    recoverssm_context: "KDARecoverSSMCommitContext | None" = field(
        default=None, repr=False, compare=False
    )

    commit_recoverssm_state = KimiK3KDAMetadata.commit_recoverssm_state


class Glm5NextKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN metadata with one state slot per request and a RecoverSSM commit."""

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert vllm_config.cache_config.use_kda_recoverssm
        # The commit folds each step's accepted tokens into the checkpoint and
        # compacts the convolution window, so the next step always starts from
        # the first slot position.
        self.recoverssm_num_accepted_tokens = torch.ones(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self.recoverssm_context: KDARecoverSSMCommitContext | None = None

    def _get_recoverssm_context(self) -> "KDARecoverSSMCommitContext":
        if self.recoverssm_context is None:
            from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
                KDARecoverSSMCommitContext,
            )

            forward_context = self.vllm_config.compilation_config.static_forward_context
            self.recoverssm_context = KDARecoverSSMCommitContext.create(
                [
                    KDARecoverSSMLayerView.of(forward_context[name])
                    for name in self.layer_names
                ],
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self.recoverssm_context

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> Glm5NextKDAMetadata:
        m = common_attn_metadata
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            # With a single checkpoint slot, every decode row has to take the
            # verify path; rows longer than the draft window cannot.
            assert m.is_prefilling is not None
            assert m.is_prefilling.device.type == "cpu"
            query_lens_cpu = m.query_start_loc_cpu.diff()
            num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu.clone()
            active_decode = (~m.is_prefilling) & (query_lens_cpu > 0)
            num_decode_draft_tokens_cpu[
                active_decode & (num_decode_draft_tokens_cpu < 0)
            ] = 0
            num_decode_draft_tokens_cpu[query_lens_cpu > self.num_spec + 1] = -1

        metadata = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        attrs = {f.name: getattr(metadata, f.name) for f in fields(metadata)}
        if metadata.num_spec_decodes == 0:
            return Glm5NextKDAMetadata(**attrs)

        assert metadata.num_accepted_tokens is not None
        assert metadata.spec_state_indices_tensor is not None
        assert metadata.spec_query_start_loc is not None
        attrs["num_accepted_tokens"] = self.recoverssm_num_accepted_tokens[
            : metadata.num_accepted_tokens.shape[0]
        ]
        request_indices = None
        if metadata.num_prefills > 0:
            assert metadata.spec_sequence_masks_cpu is not None
            request_indices = async_tensor_h2d(
                metadata.spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],
                dtype=torch.int32,
                device=m.query_start_loc.device,
            )
        align = None
        if self.kv_cache_spec.mamba_cache_mode == "align":
            align = KDARecoverSSMAlignMetadata(
                block_table=m.block_table_tensor,
                num_computed_tokens=m.compute_num_computed_tokens(),
                block_size=self.kv_cache_spec.block_size,
            )
        attrs["recoverssm_commit"] = KDARecoverSSMCommitMetadata(
            state_indices=metadata.spec_state_indices_tensor,
            query_start_loc=metadata.spec_query_start_loc,
            request_indices=request_indices,
            align=align,
        )
        attrs["recoverssm_context"] = self._get_recoverssm_context()
        return Glm5NextKDAMetadata(**attrs)


class Glm5NextKDAAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KDA"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDAMetadataBuilder]:
        return Glm5NextKDAMetadataBuilder
