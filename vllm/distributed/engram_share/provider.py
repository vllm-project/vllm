# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface between Engram models and an external table reader."""

import importlib
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch


class EngramTable(Protocol):
    """The table and local head range needed to service a lookup."""

    dim: int
    block_size: int
    num_embeddings: int
    head_start: int
    part_n_hash_cols: int
    vocab_start_idx: int
    vocab_end_idx: int
    rdma_layer_id: int | None
    rdma_full_decode_graph: bool


class EngramReader(Protocol):
    def lookup(
        self,
        table: EngramTable,
        indices: "torch.Tensor",
        out: "torch.Tensor",
        background: bool = False,
    ) -> None: ...

    def close(self) -> None: ...


class EngramProvider(Protocol):
    def get_reader(self, checkpoint_path: str | None = None) -> EngramReader: ...

    def close_reader(self) -> None: ...


def get_provider(module_name: str) -> EngramProvider:
    """Load a provider module; importing it must not create a reader."""
    return cast(EngramProvider, importlib.import_module(module_name))
