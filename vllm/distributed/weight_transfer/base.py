# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch

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

    _: KW_ONLY
    update_kind: Literal["dense", "sparse_flat"] = "dense"
    """Weight update format."""
    num_updates_list: list[int] | None = None
    """Number of sparse entries to receive for each parameter in ``names``."""

    def __post_init__(self) -> None:
        if self.update_kind not in ("dense", "sparse_flat"):
            raise ValueError(f"Unsupported update_kind: {self.update_kind}")
        if self.update_kind == "dense":
            if self.num_updates_list is not None:
                raise ValueError(
                    "Sparse metadata is only supported for `update_kind='sparse_flat'`"
                )
            return

        if self.num_updates_list is None:
            raise ValueError("`num_updates_list` is required for sparse updates")
        if len(self.num_updates_list) == 0:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        if any(num_updates < 0 for num_updates in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")

        names = getattr(self, "names", None)
        if names is not None and len(self.num_updates_list) != len(names):
            raise ValueError(
                f"`num_updates_list` should be of the same size as `names`: "
                f"got {len(self.num_updates_list)} and {len(names)}"
            )


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing parameter."""

    name: str
    indices: torch.Tensor
    values: torch.Tensor


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
    implementation, allowing different backends (NCCL, CUDA IPC[TODO], RDMA[TODO]) to be
    plugged in.

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
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
            model: The local model instance which will receive the weights
        """
        self.config = config
        self.parallel_config = parallel_config
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
    def receive_weights(
        self,
        update_info: TUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """
        Receive weights from the trainer and load them incrementally.

        Args:
            update_info: Backend-specific update info containing parameter metadata
                        and any backend-specific data
            load_weights: Callable that loads weights into the model. Called
                         incrementally for each weight to avoid OOM.
        """
        raise NotImplementedError

    def receive_sparse_weights(
        self,
        update_info: TUpdateInfo,
        apply_patches: Callable[[list[SparseWeightPatch]], None],
    ) -> None:
        """Receive sparse weight patches from the trainer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sparse weight updates"
        )

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
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """
        Send weights from trainer to inference workers.

        This is a static method that can be called from the trainer process
        to send weights to all inference workers.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples.
                     The tensors should be on the appropriate device for the backend.
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

    @staticmethod
    def trainer_send_sparse_weights(
        _iterator: Iterator[SparseWeightPatch],
        _trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Send sparse weight patches from trainer to inference workers."""
        raise NotImplementedError("Sparse weight updates are not supported")
