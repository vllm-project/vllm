# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLTrainerInitInfo,
    NCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
)

# NCCLWeightTransferInitInfo / NCCLTrainerInitInfo are re-exported here for
# convenience; their canonical home is nccl_common.
__all__ = [
    "NCCLWeightTransferInitInfo",
    "NCCLTrainerInitInfo",
    "NCCLWeightTransferUpdateInfo",
    "NCCLWeightTransferEngine",
    "NCCLTrainerWeightTransferEngine",
]


@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the dense NCCL weight transfer backend.

    Whether the transfer is packed (and the buffer geometry) is a must-agree
    wire param carried on the init info (`NCCLTrainerInitInfo` /
    `NCCLWeightTransferInitInfo`), not here; this carries only the per-round
    parameter metadata.
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self):
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )


class NCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer dense
    checkpoint-format weights from the trainer (rank 0) to all inference workers
    in a process group. Received weights are loaded via the model's
    `load_weights` using the layerwise reload lifecycle.
    """

    # Define backend-specific dataclass types
    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyNcclCommunicator | None = None
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """
        Initialize NCCL process group with the trainer and record the
        trainer-supplied wire params so the worker decodes exactly as the
        trainer encodes.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, world size, and the packed wire params
        """
        self.packed = init_info.packed
        self.packed_buffer_size_bytes = init_info.packed_buffer_size_bytes
        self.packed_num_buffers = init_info.packed_num_buffers
        self.model_update_group = worker_init_process_group(
            init_info, self.parallel_config
        )

    def start_weight_update(self) -> None:
        """Initialize layerwise reloading for the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        """
        Receive weights from trainer via NCCL broadcast.

        Whether to use packed broadcasting (and the buffer geometry) is read
        from `self.packed` / `self.packed_*`, set at the init handshake from the
        trainer's init info, so it is guaranteed to match how the trainer
        encoded.

        Args:
            update_info: NCCL update info containing parameter names, dtypes,
                        and shapes
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        if self.packed:
            # Build iterator of (name, (shape, dtype)) from update_info
            def state_dict_info_iterator():
                for name, dtype_name, shape in zip(
                    update_info.names, update_info.dtype_names, update_info.shapes
                ):
                    dtype = getattr(torch, dtype_name)
                    yield (name, (shape, dtype))

            packed_nccl_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self.model.load_weights,
                buffer_size_bytes=self.packed_buffer_size_bytes,
                num_buffers=self.packed_num_buffers,
                device=self.device,
            )
        else:
            # Use simple one-by-one broadcasting
            for name, dtype_name, shape in zip(
                update_info.names, update_info.dtype_names, update_info.shapes
            ):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device=self.device)
                self.model_update_group.broadcast(
                    weight, src=0, stream=torch.cuda.current_stream()
                )
                self.model.load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Clean up the communicator by removing the reference
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(*args: Any, **kwargs: Any) -> None:
        """Removed. Use the stateful `NCCLTrainerWeightTransferEngine` instead.

        Transitional stub kept only to satisfy the (still abstract)
        `WeightTransferEngine.trainer_send_weights`; that member is dropped from
        the worker ABC once every backend has migrated to the trainer engine.
        """
        raise NotImplementedError(
            "The static NCCL trainer path has been replaced by "
            "NCCLTrainerWeightTransferEngine. Build it via "
            "WeightTransferTrainerFactory.trainer_init(NCCLTrainerInitInfo(...), "
            "client=..., source=...) and drive it with send_weights()."
        )


class NCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[NCCLTrainerInitInfo]):
    """Trainer-side NCCL weight transfer engine.

    On the sender (rank 0) holds the NCCL communicator and drives the full
    update round trip: it runs the inference-side `update_weights` concurrently
    with the trainer-side broadcast (both rendezvous inside the same NCCL
    calls), then finishes the update. Non-sender trainer ranks hold no
    communicator; they only iterate the source to stay in the trainer-side
    collective (e.g. FSDP `full_tensor()`) and skip the client RPCs and the
    broadcast (all guarded on `is_sender`).

    `packed` / buffer sizes come from `NCCLTrainerInitInfo`; the sender
    propagates them to the worker at `trainer_init` (on the worker-side init
    info), so per-round payloads carry only parameter metadata.
    """

    init_info_cls = NCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        packed: bool = True,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.model_update_group: PyNcclCommunicator | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: NCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("NCCL trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        if not engine.is_sender:
            # Non-sender trainer ranks aren't part of the transfer NCCL group and
            # don't drive the inference side; they only participate in the
            # trainer-side gather during send_weights.
            return engine

        # Workers sit at rank_offset 1, after the single trainer sender rank 0.
        worker_init_info = NCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )

        # The inference workers block inside init_weight_transfer_engine waiting
        # for the NCCL rendezvous, so we kick that off on a side thread while we
        # open the trainer endpoint (rank 0); both sides must rendezvous together.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine, asdict(worker_init_info)
            )
            engine.model_update_group = trainer_init(init_info)
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self) -> None:
        assert self.source is not None  # guaranteed by trainer_init / __init__
        source = self.source

        # Metadata is declared without gathering. For Megatron it is itself a
        # collective, so every rank runs it; only the sender ships it.
        meta = source.metadata()

        if not self.is_sender:
            # Non-sender ranks only join the trainer-side gather collective.
            self._broadcast(source)
            return

        update_info = NCCLWeightTransferUpdateInfo(
            names=[m.name for m in meta],
            dtype_names=[str(m.dtype).split(".")[-1] for m in meta],
            shapes=[list(m.shape) for m in meta],
        )

        self.client.start_weight_update()
        # update_weights (workers receive) must run concurrently with the
        # trainer-side broadcast — both rendezvous inside the same NCCL calls.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(self.client.update_weights, asdict(update_info))
            # Cheap best-effort: if update_weights already failed (e.g. a bad
            # request rejected before any NCCL call), surface it now instead of
            # hanging in broadcast waiting for a peer that will never arrive.
            if future.done():
                future.result()
            self._broadcast(source)
            future.result()  # surface inference-side errors
        self.client.finish_weight_update()

    def _broadcast(self, source: WeightSource) -> None:
        """Iterate the source (materializing each tensor — a collective on all
        ranks) and, on the sender, broadcast from rank 0, packed or one-by-one.
        Non-sender ranks only replay the iteration to stay in the collective."""
        if self.packed:
            if self.is_sender:
                assert self.model_update_group is not None, (
                    "trainer_init() must be called before _broadcast()."
                )
                packed_nccl_broadcast_producer(
                    iterator=iter(source),
                    group=self.model_update_group,
                    src=0,
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                )
            else:
                for _ in source:
                    pass
        else:
            stream = torch.cuda.current_stream()
            for _name, tensor in source:
                if self.is_sender:
                    assert self.model_update_group is not None
                    self.model_update_group.broadcast(tensor, src=0, stream=stream)

    def shutdown(self) -> None:
        self.model_update_group = None
