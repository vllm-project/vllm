# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray Direct Transport (RDT) based weight transfer engine.

The RDT backend uses Ray's native tensor transport (with NIXL under the hood)
to move weights from a trainer Ray actor directly to vLLM inference workers.

Unlike the NCCL backend, no out-of-band process group is set up: workers
resolve the trainer actor by name and pull each weight via an RDT-tagged actor
method. See ``docs/source/serving/rdt_weight_transfer.md`` (TODO) for an
overview; the canonical example lives at ``examples/rl/rlhf_rdt.py``.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class RDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the RDT weight transfer backend."""

    trainer_actor_name: str
    """Name of the trainer Ray actor (set via ``.options(name=...)``).
    Workers resolve it via ``ray.get_actor(trainer_actor_name, namespace=...)``."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor lives in. If None, uses the
    current Ray namespace (typically the anonymous namespace created by
    ``ray.init()`` when the trainer and LLM share the same driver)."""

    produce_method_name: str = "rdt_produce_weight"
    """Name of the method on the trainer actor that returns a weight tensor by
    name. The method must be decorated with
    ``@ray.method(tensor_transport="nixl")`` to enable Direct Transport.
    Signature: ``(name: str) -> torch.Tensor``."""


@dataclass
class RDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the RDT weight transfer backend."""

    names: list[str]
    """Parameter names to fetch from the trainer, in fetch order."""

    dtype_names: list[str]
    """Expected dtype for each parameter (e.g. 'float32', 'bfloat16').
    Carried for parity with the NCCL backend; the actual dtype comes from
    the trainer's tensor."""

    shapes: list[list[int]]
    """Expected shape for each parameter. Carried for parity with the NCCL
    backend; the actual shape comes from the trainer's tensor."""

    def __post_init__(self) -> None:
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


class RDTWeightTransferEngine(
    WeightTransferEngine[RDTWeightTransferInitInfo, RDTWeightTransferUpdateInfo]
):
    """Weight transfer engine using Ray Direct Transport (RDT) with NIXL.

    Pull-based: each inference worker is a Ray actor that fetches weights
    directly from a named trainer Ray actor via an RDT-tagged actor method.
    NIXL handles the device-to-device transport; no collective group setup is
    required.

    Requirements:
        - vLLM workers must be Ray actors. Configure the LLM with
          ``distributed_executor_backend="ray"``.
        - The trainer must be a Ray actor created with
          ``.options(name=trainer_actor_name)`` that exposes a method decorated
          with ``@ray.method(tensor_transport="nixl")`` named
          ``produce_method_name`` returning a ``torch.Tensor`` when called with
          a parameter name.
        - NIXL must be installed in the environment shared by the trainer and
          worker actors (``pip install nixl``).
    """

    init_info_cls = RDTWeightTransferInitInfo
    update_info_cls = RDTWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._trainer_actor: Any | None = None
        self._produce_method: Any | None = None

    def init_transfer_engine(self, init_info: RDTWeightTransferInitInfo) -> None:
        """Resolve the trainer actor by name and bind its RDT producer method.

        This is the only setup needed for the NIXL transport — no collective
        group, no host/port handshake. ``ray.get_actor`` is a name lookup
        against Ray's GCS, so this works from any process connected to the
        same Ray cluster.
        """
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "Ray is required for the 'rdt' weight transfer backend. "
                "Install Ray and ensure workers run as Ray actors "
                "(distributed_executor_backend='ray')."
            ) from e

        try:
            self._trainer_actor = ray.get_actor(
                init_info.trainer_actor_name,
                namespace=init_info.trainer_actor_namespace,
            )
        except ValueError as e:
            # ray.get_actor raises ValueError when the named actor does not
            # exist in this namespace (also covers the "this process is not
            # connected to the Ray cluster" case for workers).
            raise RuntimeError(
                f"RDT weight transfer engine could not find trainer actor "
                f"named {init_info.trainer_actor_name!r} (namespace="
                f"{init_info.trainer_actor_namespace!r}). Ensure (1) the "
                f"trainer is created with .options(name=...), (2) this worker "
                f"runs as a Ray actor (set "
                f"distributed_executor_backend='ray' on the LLM), and (3) "
                f"namespaces match."
            ) from e

        # Ray bug workaround (as of ray==2.51.1): `ray.get_actor` reconstructs
        # the destination actor handle in this process via the GCS, and the
        # actor-level `_ray_enable_tensor_transport` flag does NOT round-trip
        # through that path -- it always comes back False, even when the
        # target actor was created with `@ray.remote(enable_tensor_transport=
        # True)`. The per-method tensor_transport map IS preserved, so without
        # this patch the `.remote()` dispatch hits the guard at
        # ray/actor.py:829 (`if not self._actor._ray_enable_tensor_transport`)
        # and raises despite the trainer being correctly configured.
        # Force-set the flag here so the NIXL-tagged producer method actually
        # dispatches via NIXL.
        self._trainer_actor._ray_enable_tensor_transport = True

        self._produce_method = getattr(
            self._trainer_actor, init_info.produce_method_name
        )
        logger.info(
            "RDT weight transfer engine bound to trainer actor %r (method %r)",
            init_info.trainer_actor_name,
            init_info.produce_method_name,
        )

    def receive_weights(
        self,
        update_info: RDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Stream-pull each weight from the trainer and fold it into the model.

        One tensor is in flight at a time: the worker calls the trainer's
        RDT-tagged method to get an ``ObjectRef``, ``ray.get``s it (NIXL
        transfers the tensor to this worker's GPU), hands it to
        ``load_weights`` (which copies it into the model's pre-allocated
        parameter memory), then drops the local reference before fetching the
        next.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "RDT weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        import ray

        for name in update_info.names:
            ref = self._produce_method.remote(name)
            tensor = ray.get(ref)
            load_weights([(name, tensor)])
            del tensor
            del ref

    def shutdown(self) -> None:
        self._trainer_actor = None
        self._produce_method = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """No-op for the pull-based RDT backend.

        With the NCCL backend the trainer drives a broadcast from inside this
        method. With RDT the workers initiate the transfer themselves by
        calling the trainer actor's ``@ray.method(tensor_transport="nixl")``
        accessor — the trainer does not need to push anything from here.
        Method retained to satisfy the abstract base class.
        """
        del iterator, trainer_args
