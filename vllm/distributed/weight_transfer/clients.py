# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in `VLLMWeightSyncClient` implementations.

These adapt the inference engine's weight-sync control plane to concrete
transports. A `TrainerWeightTransferEngine` takes one of these (or any object
with the same four methods — the protocol is structural) and drives the full
handshake through it.

Imports of `ray` / `requests` are deferred to call time so this module is
importable without those packages installed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.actor import ActorHandle


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


class HTTPVLLMWeightSyncClient:
    """Talks to a vLLM server over the RLHF HTTP routes.

    Mirrors `vllm/entrypoints/serve/dev/rlhf/api_router.py`:
    `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`,
    `/finish_weight_update`.
    """

    def __init__(self, base_url: str, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, json: dict[str, Any] | None = None) -> None:
        import requests

        response = requests.post(
            f"{self.base_url}/{path}", json=json, timeout=self.timeout
        )
        response.raise_for_status()

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._post("init_weight_transfer_engine", {"init_info": init_info})

    def start_weight_update(self) -> None:
        self._post("start_weight_update")

    def update_weights(self, update_info: dict[str, Any]) -> None:
        self._post(
            "update_weights", {"update_info": _json_safe_update_info(update_info)}
        )

    def finish_weight_update(self) -> None:
        self._post("finish_weight_update")


class RayVLLMWeightSyncClient:
    """Talks to one or more vLLM `AsyncLLM`/`LLM` Ray actors.

    Each call fans out to every handle and blocks on all of them, so a
    multi-actor (e.g. multi-DP) deployment is driven as one unit.
    """

    def __init__(self, handle: "ActorHandle | list[ActorHandle]") -> None:
        self.handles = handle if isinstance(handle, list) else [handle]

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        import ray

        ray.get(
            [
                h.init_weight_transfer_engine.remote({"init_info": init_info})
                for h in self.handles
            ]
        )

    def start_weight_update(self) -> None:
        import ray

        ray.get([h.start_weight_update.remote() for h in self.handles])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        import ray

        ray.get(
            [
                h.update_weights.remote({"update_info": update_info})
                for h in self.handles
            ]
        )

    def finish_weight_update(self) -> None:
        import ray

        ray.get([h.finish_weight_update.remote() for h in self.handles])
