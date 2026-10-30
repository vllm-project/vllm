# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, fields
from typing import Any

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class Mamba1Spec(MambaSpec):
    """Conv state ``[x, conv]`` and temporal state ``[x, state]``, both
    sharded across TP along ``x``."""

    def _tp_segments(self) -> tuple[tuple[int, ...], ...] | None:
        if len(self.shapes) != 2:
            return None
        conv, temporal = self.shapes
        return (conv[0],), (temporal[0],)


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def customize_mamba_spec(cls, spec: MambaSpec) -> MambaSpec:
        # x leads the conv state only in the (dim, state_len) layout.
        if not is_conv_state_dim_first():
            return spec
        return Mamba1Spec(**{f.name: getattr(spec, f.name) for f in fields(spec)})


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    pass


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        return self._compute_common_metadata(common_attn_metadata)
