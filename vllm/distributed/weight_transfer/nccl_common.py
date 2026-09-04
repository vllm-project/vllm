# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared NCCL initialization helpers for weight transfer engines.

The dense (`NCCLWeightTransferEngine`) and sparse
(`SparseNCCLWeightTransferEngine`) backends are independent engines that share
*only* their process-group initialization. That common logic lives here so the
sparse engine does not have to subclass the dense one.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

import pybase64 as base64
import torch

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCL_UNIQUE_ID_BYTES,
)
from vllm.distributed.weight_transfer.base import WeightTransferInitInfo
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)


def decode_nccl_unique_id(
    *,
    master_address: str | None,
    master_port: int | None,
    nccl_unique_id_b64: str | None,
    ctx: str,
) -> bytes | None:
    """Validate the rendezvous mode and decode a pre-shared unique id."""
    has_uid = bool(nccl_unique_id_b64)
    has_addr = master_address is not None
    has_port = master_port is not None

    if has_uid and (has_addr or has_port):
        raise ValueError(
            f"{ctx}: pass nccl_unique_id_b64 OR master_address/master_port, not both"
        )
    if not has_uid:
        if not (has_addr and has_port):
            raise ValueError(
                f"{ctx}: need nccl_unique_id_b64, or both master_address and "
                f"master_port (got master_address={master_address!r}, "
                f"master_port={master_port!r})"
            )
        return None
    try:
        decoded = base64.b64decode(nccl_unique_id_b64, validate=True)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"{ctx}: nccl_unique_id_b64 is not valid standard base64 "
            f"(URL-safe base64 is not accepted)"
        ) from e
    if len(decoded) != NCCL_UNIQUE_ID_BYTES:
        raise ValueError(
            f"{ctx}: nccl_unique_id_b64 must decode to {NCCL_UNIQUE_ID_BYTES} "
            f"bytes, got {len(decoded)}"
        )
    return decoded


@dataclass(kw_only=True)
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side initialization info for NCCL-based weight transfer backends.

    Keyword-only (`kw_only`): adding the optional `nccl_unique_id_b64` field
    means the rendezvous fields can no longer keep a fixed positional slot, so a
    stale positional call fails loudly instead of silently swapping arguments.

    Provide exactly one rendezvous mode:

    * ``master_address`` + ``master_port`` -- TCPStore / ``StatelessProcessGroup``
      rendezvous (requires torch on every rank, including the trainer), or
    * ``nccl_unique_id_b64`` -- standard (RFC 4648, *not* URL-safe) base64 of the
      128 raw bytes from ``ncclGetUniqueId``, for torch-free trainers (e.g. JAX)
      that mint the unique id out of band and share it (over HTTP, etc.). Note a
      JAX peer must use ``base64.b64encode``, not ``urlsafe_b64encode``.

    On the unique-id path all ranks must enter init concurrently (there is no
    store barrier), and every peer must honor the warm-up handshake: the
    worker's communicator issues a one-element ``all_reduce`` immediately after
    ``ncclCommInitRank`` (see ``PyNcclCommunicator.from_unique_id_bytes``), so a
    foreign peer must issue a matching one-element ``all_reduce`` before any
    other collective or all ranks deadlock.
    """

    master_address: str | None = None
    master_port: int | None = None
    rank_offset: int
    world_size: int
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    packed: bool = False
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS

    def __post_init__(self) -> None:
        _ = self.nccl_unique_id_bytes

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        return decode_nccl_unique_id(
            master_address=self.master_address,
            master_port=self.master_port,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="NCCLWeightTransferInitInfo",
        )


def worker_init_payload(init_info: NCCLWeightTransferInitInfo) -> dict:
    """Serialize a worker init info for `init_weight_transfer_engine`, dropping
    the unset rendezvous field (the UID in TCP mode) so the wire payload carries
    only the mode actually in use. Shared by the dense and sparse trainer
    engines so the two cannot drift."""
    return {key: value for key, value in asdict(init_info).items() if value is not None}


class NCCLRendezvous(Protocol):
    """The TCP rendezvous fields `trainer_init` needs.

    Structural so each backend can keep its own trainer init info next to its
    engine (dense `NCCLTrainerInitInfo`, `SparseNCCLTrainerInitInfo`) without
    this shared module importing either.
    """

    master_address: str
    master_port: int
    world_size: int


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device,
) -> "PyNcclCommunicator":
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


def uid_init_process_group(
    nccl_unique_id_bytes: bytes,
    rank: int,
    world_size: int,
    device,
) -> "PyNcclCommunicator":
    """Join the NCCL group from pre-shared ``ncclUniqueId`` bytes.

    The torch-free rendezvous alternative to `stateless_init_process_group`: no
    TCPStore, and therefore no barrier -- every rank must enter concurrently.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator.from_unique_id_bytes(
        nccl_unique_id_bytes,
        rank=rank,
        world_size=world_size,
        device=device,
    )


def worker_init_process_group(
    init_info: NCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> "PyNcclCommunicator":
    """Create the trainer<->worker NCCL group on an inference worker.

    Computes a unique rank for this worker across all data-parallel groups and
    joins the trainer via whichever rendezvous mode `init_info` carries.
    """
    # Calculate the global rank in the trainer-worker process group.
    # Must account for data parallel to get unique ranks across all workers.
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size  # TP * PP
    rank_within_dp = parallel_config.rank

    # Unique rank across all DP groups
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset

    device = torch.accelerator.current_device_index()
    unique_id_bytes = init_info.nccl_unique_id_bytes
    if unique_id_bytes is not None:
        return uid_init_process_group(
            unique_id_bytes,
            rank,
            init_info.world_size,
            device,
        )
    # __post_init__ guarantees the TCP pair when there is no unique id.
    assert init_info.master_address is not None
    assert init_info.master_port is not None
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device=device,
    )


def trainer_init(
    init_info: NCCLRendezvous | dict,
) -> "PyNcclCommunicator":
    """
    Initialize NCCL process group for trainer-side weight transfer.

    The trainer is always rank 0 in the process group. Uses the current
    CUDA device (torch.accelerator.current_device_index()).

    Args:
        init_info: Any object carrying the `NCCLRendezvous` fields (a trainer or
            worker NCCL init info), or a dict with keys:
            - master_address: str
            - master_port: int
            - world_size: int

    Returns:
        PyNcclCommunicator for weight transfer.
    """
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
        master_address = init_info.master_address
        master_port = init_info.master_port
        world_size = init_info.world_size

    # Trainer is always rank 0
    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        master_address,
        master_port,
        0,
        world_size,
        device,
    )
