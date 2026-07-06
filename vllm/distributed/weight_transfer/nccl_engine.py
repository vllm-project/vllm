# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.weight_transfer import NCCLWeightTransferConfig
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

    Static wire params (`packed`, buffer sizes) now live on
    `NCCLWeightTransferConfig`; this carries only the per-round parameter
    metadata.
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

    # Narrow the base `config` type: the dense NCCL engine always carries the
    # static wire params (`packed`, buffer sizes) that live on the subclass.
    config: NCCLWeightTransferConfig

    def __init__(
        self,
        config: NCCLWeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyNcclCommunicator | None = None

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """
        Initialize NCCL process group with the trainer.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, and world size
        """
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

        Whether to use packed broadcasting (and the buffer geometry) is read from
        `self.config`, so it is guaranteed to match the trainer's config.

        Args:
            update_info: NCCL update info containing parameter names, dtypes,
                        and shapes
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        if self.config.packed:
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
                buffer_size_bytes=self.config.packed_buffer_size_bytes,
                num_buffers=self.config.packed_num_buffers,
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


class NCCLTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[NCCLWeightTransferConfig, NCCLTrainerInitInfo]
):
    """Trainer-side NCCL weight transfer engine.

    On the sender (rank 0) holds the NCCL communicator and drives the full
    update round trip: it runs the inference-side `update_weights` concurrently
    with the trainer-side broadcast (both rendezvous inside the same NCCL
    calls), then finishes the update. Non-sender trainer ranks hold no
    communicator; they only iterate the source to stay in the trainer-side
    collective (e.g. FSDP `full_tensor()`) and skip the client RPCs and the
    broadcast (all guarded on `is_sender`).
    """

    init_info_cls = NCCLTrainerInitInfo
    config_cls = NCCLWeightTransferConfig

    def __init__(
        self,
        config: NCCLWeightTransferConfig,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
    ) -> None:
        super().__init__(config, client=client, source=source, is_sender=is_sender)
        self.model_update_group: PyNcclCommunicator | None = None

    @classmethod
    def trainer_init(
        cls,
        config: NCCLWeightTransferConfig,
        init_info: NCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
    ) -> Self:
        is_sender = init_info.is_sender
        engine = cls(config, client=client, source=source, is_sender=is_sender)
        if not is_sender:
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
        if self.config.packed:
            if self.is_sender:
                assert self.model_update_group is not None, (
                    "trainer_init() must be called before _broadcast()."
                )
                packed_nccl_broadcast_producer(
                    iterator=iter(source),
                    group=self.model_update_group,
                    src=0,
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=self.config.packed_buffer_size_bytes,
                    num_buffers=self.config.packed_num_buffers,
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
