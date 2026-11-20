# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in `VLLMWeightSyncClient` implementations.

These adapt the inference engine's weight-sync control plane to concrete
transports. A `TrainerWeightTransferEngine` takes one of these (or any object
with the same methods — the protocol is structural) and drives the full
handshake through it.

Imports of `ray` / `requests` are deferred to call time so this module is
importable without those packages installed.
"""

from typing import TYPE_CHECKING, Any

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdatePayload,
    WeightTransferUpdateRequest,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from vllm.v1.engine import PauseMode

logger = init_logger(__name__)


def _json_safe_update_info(update_info: dict[str, Any]) -> dict[str, Any]:
    """Make an update_info dict JSON-serializable for HTTP transport.

    CUDA IPC handles (`ipc_handles`) are tuples of non-JSON-native objects, so
    over HTTP they are pickled+base64-encoded into `ipc_handles_pickled` (which
    the worker auto-deserializes when `VLLM_ALLOW_INSECURE_SERIALIZATION=1`).
    Other backends (NCCL) carry only JSON-native metadata and pass through
    unchanged. Mirrors the old IPC `_do_send` HTTP branch.
    """
    ipc_handles = update_info.get("ipc_handles")
    if ipc_handles is None:
        return update_info

    import pickle

    import pybase64 as base64

    out = {k: v for k, v in update_info.items() if k != "ipc_handles"}
    out["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(ipc_handles)).decode(
        "utf-8"
    )
    return out


def _json_safe_update_payload(
    update_info: WeightTransferUpdatePayload,
) -> WeightTransferUpdatePayload:
    if isinstance(update_info, list):
        return [_json_safe_update_info(info) for info in update_info]
    return _json_safe_update_info(update_info)


class HTTPVLLMWeightSyncClient:
    """Talks to a vLLM server over the RLHF HTTP routes.

    Mirrors `vllm/entrypoints/rl/online/api_router.py`:
    `/init_weight_transfer_engine`, `/stage_weights`, `/start_weight_update`,
    `/update_weights`, `/finish_weight_update`,
    `/shutdown_weight_transfer_engine`, `/pause`, `/resume`.
    """

    def __init__(self, base_url: str, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        import requests

        response = requests.post(
            f"{self.base_url}/{path}", json=json, params=params, timeout=self.timeout
        )
        response.raise_for_status()

    def pause_generation(
        self, mode: "PauseMode" = "keep", clear_cache: bool = True
    ) -> None:
        self._post(
            "pause", params={"mode": mode, "clear_cache": str(clear_cache).lower()}
        )

    def resume_generation(self) -> None:
        self._post("resume")

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._post("init_weight_transfer_engine", {"init_info": init_info})

    def shutdown_weight_transfer_engine(self) -> None:
        self._post("shutdown_weight_transfer_engine")

    def stage_weights(self, update_info: WeightTransferUpdatePayload) -> None:
        self._post(
            "stage_weights", {"update_info": _json_safe_update_payload(update_info)}
        )

    def start_weight_update(self) -> None:
        self._post("start_weight_update")

    def update_weights(self, update_info: WeightTransferUpdatePayload) -> None:
        self._post(
            "update_weights", {"update_info": _json_safe_update_payload(update_info)}
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        json = (
            {"weight_version": weight_version} if weight_version is not None else None
        )
        self._post("finish_weight_update", json)


class RayVLLMWeightSyncClient:
    """Talks to one or more vLLM `AsyncLLM`/`LLM` Ray actors.

    Each call fans out to every handle and blocks on all of them, so a
    multi-actor (e.g. multi-DP) deployment is driven as one unit. Pause calls
    skip actors without them (e.g. offline `LLM`, which serializes calls anyway),
    with a warning.
    """

    def __init__(self, handle: "ActorHandle | list[ActorHandle]") -> None:
        self.handles = handle if isinstance(handle, list) else [handle]

    def _pausable(self) -> list["ActorHandle"]:
        methods = ("pause_generation", "resume_generation")
        pausable = [h for h in self.handles if all(hasattr(h, m) for m in methods)]
        if len(pausable) < len(self.handles):
            logger.warning_once(
                "%d of %d Ray actors lack pause_generation/resume_generation; "
                "they are not paused during weight updates.",
                len(self.handles) - len(pausable),
                len(self.handles),
            )
        return pausable

    def pause_generation(
        self, mode: "PauseMode" = "keep", clear_cache: bool = True
    ) -> None:
        import ray

        ray.get(
            [
                h.pause_generation.remote(mode=mode, clear_cache=clear_cache)
                for h in self._pausable()
            ]
        )

    def resume_generation(self) -> None:
        import ray

        ray.get([h.resume_generation.remote() for h in self._pausable()])

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        import ray

        request = WeightTransferInitRequest(init_info=init_info)
        ray.get([h.init_weight_transfer_engine.remote(request) for h in self.handles])

    def shutdown_weight_transfer_engine(self) -> None:
        import ray

        ray.get([h.shutdown_weight_transfer_engine.remote() for h in self.handles])

    def stage_weights(self, update_info: WeightTransferUpdatePayload) -> None:
        import ray

        request = WeightTransferUpdateRequest(update_info=update_info)
        ray.get([h.stage_weights.remote(request) for h in self.handles])

    def start_weight_update(self) -> None:
        import ray

        ray.get([h.start_weight_update.remote() for h in self.handles])

    def update_weights(self, update_info: WeightTransferUpdatePayload) -> None:
        import ray

        request = WeightTransferUpdateRequest(update_info=update_info)
        ray.get([h.update_weights.remote(request) for h in self.handles])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        import ray

        ray.get([h.finish_weight_update.remote() for h in self.handles])
        if weight_version is not None:
            ray.get(
                [h.update_weight_version.remote(weight_version) for h in self.handles]
            )
