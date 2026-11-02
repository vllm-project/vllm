# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-chain recording for the sharded-RDT backend.

The consumer asks the trainer for the exact slice a worker consumes, described as
an op chain replayed on the trainer's live tensor. ``FakeRDTTensor`` (the reload
core's ``TraceTensor``) builds the chain by intercepting the model's own weight
loaders; ``copy_`` is its data sink, ``BakeSink`` — the dry run records how each
slice would be fetched and where it lands.

Chains are built against meta tensors, so nothing here touches Ray or a GPU.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.model_executor.model_loader.reload.trace import (
    OpChain,
    TraceSink,
    TraceTensor,
    UnsupportedTraceOp,
)

# The chain serves as the trainer-side fetch key.
FetchKey = tuple[str, OpChain]


@dataclass
class _Scatter:
    """One recorded scatter: pull ``src``, copy it into ``layer``'s
    ``param_name`` at the recorded strided region. The bake's output unit; see
    the engine module's Data flow.

    Self-contained so replay needs no lookups. Destination geometry is read off
    the meta view and rebuilt each sync as ``as_strided(shape, stride, offset)``
    -- ``layer`` is resolved to a param at replay time, never baked, since every
    sync re-materializes fresh tensors.

    ``shape`` and ``dtype`` also size the slice ON THE WIRE, so ``dtype`` is the
    PRODUCED dtype (the fake's after its op chain), not the source name's:
    ``view(dtype)`` is allowlisted, and taking the source's would size the slice
    with the wrong itemsize and carve the packed blob differently on the two
    sides.
    """

    layer: Any
    param_name: str
    src: FetchKey
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


def _meta_copy_(dest: torch.Tensor, src: TraceTensor) -> torch.Tensor:
    """Fire ``dest.copy_`` from a zero-storage meta source of ``src``'s geometry.

    Moves no data but still counts against the layer's loaded numel, which drives
    ``_layerwise_process``; skipping it would leave the layer looking unloaded
    forever.

    A non-meta ``dest`` is one the layerwise reload never moved there (the
    ``SKIP_LOAD_TENSORS`` set, e.g. GLM's router bias). Torch forbids
    ``real.copy_(meta)``, and skipping the count is consistent -- ``get_layer_size``
    excludes the same set. The caller still RECORDS the copy, so the replay writes
    the real param.
    """
    if not dest.is_meta:
        return dest
    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
    with torch._C.DisableTorchFunctionSubclass():
        return dest.copy_(meta_src)


@dataclass
class BakeSink(TraceSink):
    """``copy_`` sink for the dry-run bake: record how each slice would be fetched
    and where it would land, and move nothing.

    ``_install_recording_stamps`` sets ``current = (leaf_module, param_name)``
    around each loader so ``accept_copy`` can attribute the copy. An unstamped
    ``copy_`` cannot be attributed and stays unrecorded, so its module fails the
    coverage gate and takes the plain load. ``copies_by_layer`` is keyed by module
    object, so iterating it yields each leaf module once.
    """

    copies_by_layer: "dict[Any, list[_Scatter | None]]" = field(
        default_factory=lambda: defaultdict(list)
    )
    current: "tuple[Any, str] | None" = None
    # Source names whose ``copy_`` fired during the bake. Names not here never
    # moved data (e.g. experts owned by another EP rank) and are skipped; names
    # here but unbaked fail the plan build.
    copied_names: "set[str]" = field(default_factory=set)

    def accept_copy(self, dest: torch.Tensor, src: TraceTensor) -> torch.Tensor:
        self.copied_names.add(src._name)
        if self.current is not None:
            layer, param_name = self.current
            if dest.numel() != src.numel():
                # The packed layout sizes each slice from the DESTINATION shape,
                # while the producer packs what the replayed chain produces. A
                # broadcasting copy_ makes those two lengths differ, and neither
                # side can notice: the consumer's buffer view is exactly
                # prod(dest.shape) elements, so it reshapes cleanly over bytes
                # the producer laid out at different offsets. Refuse at bake.
                raise UnsupportedTraceOp(
                    f"FakeRDTTensor: broadcasting copy_ into {param_name!r} of "
                    f"{type(layer).__name__} ({src.numel()} elements from "
                    f"{src._name!r} into {dest.numel()}). The sharded RDT "
                    "backend sends exactly the slice the loader asks for, so "
                    "the source and destination must have equal numel."
                )
            self.copies_by_layer[layer].append(
                _Scatter(
                    layer=layer,
                    param_name=param_name,
                    src=src._key(),
                    offset=dest.storage_offset(),
                    shape=tuple(dest.shape),
                    stride=tuple(dest.stride()),
                    dtype=src.dtype,
                )
            )
        return _meta_copy_(dest, src)


# The tracer moved into the reload core; these names keep this module's API.
# TODO(E1): remove with the dry run.
FakeRDTTensor = TraceTensor
_UnsupportedFakeOp = UnsupportedTraceOp
