# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only observation of real encoder execution and Store transfers."""

import hashlib
import os
import threading
import uuid
from typing import Any, cast

import requests


def _digest(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _replicas(store, keys):
    descriptors = store.batch_get_replica_desc(keys)
    result = {}
    for key in keys:
        result[key] = [
            "memory"
            if replica.is_memory_replica()
            else "local_disk"
            if replica.is_local_disk_replica()
            else "unknown"
            for replica in descriptors[key]
        ]
    return result


def _admin_replicas(keys):
    """Observe replicas without granting the target a new Store read lease."""
    if not keys:
        return {}
    assert not any("," in key for key in keys), (
        "Admin batch API uses comma-separated keys"
    )
    url = os.environ.get("SSD_E2E_MASTER_ADMIN_URL", "http://127.0.0.1:9003")
    response = requests.get(
        url + "/batch_query_keys", params={"keys": ",".join(keys)}, timeout=5
    )
    assert response.ok, (
        f"Admin observation failed: {response.status_code} {response.text}"
    )
    payload = response.json()
    assert payload["success"], payload
    result = {}
    for key in keys:
        item = payload["data"][key]
        if not item["ok"]:
            raise RuntimeError(f"Store object disappeared: {key}: {item.get('error')}")
        kinds = []
        for field, kind in (
            ("values", "memory"),
            ("local_disk_values", "local_disk"),
            ("disk_values", "disk"),
            ("nof_values", "unknown"),
        ):
            kinds.extend([kind] * len(item.get(field) or []))
        result[key] = kinds
    return result


class SSDTestWorkerExtension:
    model_runner: Any

    def ssd_test_arm(self):
        from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
            store_client as codec,
        )
        from vllm.distributed.ec_transfer.ec_transfer_state import get_ec_transfer

        if hasattr(self, "_ssd_test_state"):
            raise RuntimeError("Observation must be armed once on a fresh Encoder")
        from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
            ECMooncakeConnector,
        )

        connector = cast(ECMooncakeConnector, get_ec_transfer())
        assert connector._worker is not None
        backend = connector._worker._output_store
        assert backend is not None
        state: dict[str, Any] = {
            "encoded_items": 0,
            "published": {},
            "loaded": {},
            "get_replicas": {},
            "put_errors": {},
            "published_sizes": {},
        }
        self._ssd_test_state = state
        self._ssd_test_lock = threading.Lock()
        self._ssd_test_backend = backend
        client: Any = backend.store_client
        original_put = client.put_tensor
        original_load = client.load_tensors
        self._ssd_test_put = original_put
        runner = self.model_runner.model_state.encoder_runner
        original_encode = runner.execute_mm_encoder

        def encode(inputs):
            outputs = original_encode(inputs)
            with self._ssd_test_lock:
                state["encoded_items"] += len(inputs)
            return outputs

        def put(key, tensor):
            try:
                digest = _digest(tensor)
                original_put(key, tensor)
            except Exception as error:
                with self._ssd_test_lock:
                    state["put_errors"][key] = repr(error)
                raise
            with self._ssd_test_lock:
                state["published"][key] = digest
                object_bytes = codec._MOONCAKE_TENSOR_HEADER.size + tensor.nbytes
                state["published_sizes"][key] = {
                    "payload_bytes": tensor.nbytes,
                    "object_bytes": object_bytes,
                    "disk_staging_bytes": codec._estimate_disk_staging_bytes(
                        object_bytes
                    ),
                }

        def load(expected, device):
            replicas = _replicas(client.store, list(expected)) if expected else {}
            outputs = original_load(expected, device)
            with self._ssd_test_lock:
                state["get_replicas"].update(replicas)
                state["loaded"].update({key: _digest(t) for key, t in outputs.items()})
            return outputs

        runner.execute_mm_encoder = encode
        client.put_tensor = put
        client.load_tensors = load
        return True

    def ssd_test_state(self):
        backend = self._ssd_test_backend
        backend.reap()
        with self._ssd_test_lock:
            state = self._ssd_test_state
            result = {
                "encoded_items": state["encoded_items"],
                "published": dict(state["published"]),
                "loaded": dict(state["loaded"]),
                "get_replicas": dict(state["get_replicas"]),
                "put_errors": dict(state["put_errors"]),
                "published_sizes": dict(state["published_sizes"]),
                "read_buffer_bytes": backend.store_client._read_buffer_bytes,
                "disk_staging_budget_bytes": (
                    backend.store_client._disk_staging_budget_bytes
                ),
            }
        with backend._pending_lock:
            result["pending_puts"] = len(backend._pending)
        result["replicas"] = _admin_replicas(list(result["published"]))
        return result

    def ssd_test_pressure(self):
        """Publish one bounded 16 MiB unrelated object to cause real eviction."""
        import torch

        from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
            store_client,
        )

        backend = self._ssd_test_backend
        tensor = torch.zeros((1024, 8192), dtype=torch.float16, device="cpu")
        try:
            self._ssd_test_put(
                f"{backend.namespace}@e2e-pressure:{uuid.uuid4().hex}", tensor
            )
        except store_client.EmbeddingStoreOperationError as error:
            return {"written_bytes": 0, "rejection": str(error)}
        return {"written_bytes": tensor.nbytes}
