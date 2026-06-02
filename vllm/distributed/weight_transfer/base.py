# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig

TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")


# Base protocols for backend-specific dataclasses
@dataclass
class WeightTransferInitInfo(ABC):  # noqa: B024
    """Base class for backend-specific initialization info."""

    pass


@dataclass
class WeightTransferUpdateInfo(ABC):  # noqa: B024
    """Base class for backend-specific weight update info."""

    pass


# API-level request classes (accept dicts for backend-agnostic serialization)
@dataclass
class WeightTransferInitRequest:
    """API-level weight transfer initialization request."""

    init_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightTransferUpdateRequest:
    """API-level weight update request."""

    update_info: dict[str, Any] = field(default_factory=dict)


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """
    Base class for weight transfer engines that handle transport of model weights
    from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the worker
    implementation, allowing different backends (NCCL, CUDA IPC, RDMA[TODO]) to be
    plugged in.

    Each engine owns its full weight-update lifecycle: `start_weight_update`,
    `update_weights`, and `finish_weight_update`. Layerwise reloading (used by
    checkpoint-format engines) is opted into per engine by running it inside
    `start_weight_update`/`finish_weight_update`. Engines that apply weights in
    place (e.g. sparse patches) leave those methods as no-ops.

    Session lifecycle state (whether an update is active) is tracked by the
    worker, not the engine, so subclasses do not need to chain to `super()` in
    their lifecycle methods.

    Subclasses should define:
        init_info_cls: Type of backend-specific initialization info
        update_info_cls: Type of backend-specific update info
    """

    # Subclasses should override these class attributes
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            vllm_config: The full vLLM config (provides parallel/model config and
                is used to set the current config when running layerwise reload)
            device: The device this worker's model lives on
            model: The local model instance which will receive the weights
        """
        self.config = config
        self.vllm_config = vllm_config
        self.parallel_config: ParallelConfig = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.model = model

    def parse_init_info(self, init_dict: dict[str, Any]) -> TInitInfo:
        """
        Construct typed init info from dict with validation.

        Args:
            init_dict: Dictionary containing backend-specific initialization parameters

        Returns:
            Typed backend-specific init info dataclass

        Raises:
            ValueError: If init_dict is invalid for this backend
        """
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid init_info for {self.__class__.__name__}: {e}"
            ) from e

    def parse_update_info(self, update_dict: dict[str, Any]) -> TUpdateInfo:
        """
        Construct typed update info from dict with validation.

        Args:
            update_dict: Dictionary containing backend-specific update parameters

        Returns:
            Typed backend-specific update info dataclass

        Raises:
            ValueError: If update_dict is invalid for this backend
        """
        try:
            return self.update_info_cls(**update_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid update_info for {self.__class__.__name__}: {e}"
            ) from e

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            init_info: Backend-specific initialization info
        """
        raise NotImplementedError

    @abstractmethod
    def start_weight_update(self) -> None:
        """
        Prepare the engine for a new weight update.

        Checkpoint-format engines initialize layerwise reloading here; engines
        that apply weights in place leave this as a no-op. Must not chain to
        `super()`.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        Checkpoint-format engines finalize layerwise reloading here; engines
        that apply weights in place leave this as a no-op. Must not chain to
        `super()`.
        """
        raise NotImplementedError

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """
        Receive one weight update chunk and load it into the model.

        This is stateless orchestration: parse the backend-specific update info,
        receive the weights, then synchronize so the new weights are visible to
        the next forward pass. Session-lifecycle bookkeeping is handled by the
        worker.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)
        # NCCL broadcast / IPC paths may be asynchronous. Synchronize here so the
        # next step uses the new weights.
        torch.accelerator.synchronize()

    @abstractmethod
    def receive_weights(self, update_info: TUpdateInfo) -> None:
        """
        Receive weights from the trainer and load them into the model.

        Implementations should load weights incrementally (one or a few at a
        time) into `self.model` to avoid OOM.

        Args:
            update_info: Backend-specific update info containing parameter metadata
                        and any backend-specific data
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        This should be called when the worker is shutting down.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """
        Send weights from trainer to inference workers.

        This is a static method that can be called from the trainer process
        to send weights to all inference workers.

        Args:
            iterator: Iterator of backend-specific items to send. Dense engines
                     iterate (name, tensor) tuples; sparse engines iterate
                     patch objects. Tensors should be on the appropriate device.
            trainer_args: Dictionary containing backend-specific arguments needed
                         to send weights. The structure depends on the backend:
                         - NCCL: Contains 'group', 'src', 'packed', etc.
                         - IPC: Contains 'mode' ('http' or 'ray'),
                                'llm_handle' (for Ray), 'url' (for HTTP), etc.

        Example:
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> engine.trainer_send_weights(param_iter, trainer_args)
        """
        raise NotImplementedError
