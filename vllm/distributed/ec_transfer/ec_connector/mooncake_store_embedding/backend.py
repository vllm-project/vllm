# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-side Mooncake Store resolver and publisher."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

from .data import TensorSpec, make_embedding_key
from .store_client import (
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
)

logger = init_logger(__name__)


@dataclass
class _StoreSave:
    tensor: torch.Tensor | None
    ready_event: torch.Event | None
    storage_ptr: int

    def release(self) -> None:
        self.tensor = None
        self.ready_event = None


class MooncakeEmbeddingStoreBackend:
    """Resolve immutable encoder outputs and publish misses asynchronously."""

    def __init__(
        self,
        store_client: MooncakeEmbeddingStoreClient,
        namespace: str,
        *,
        max_pending_items: int,
        max_pending_bytes: int,
    ) -> None:
        self.store_client = store_client
        self.namespace = namespace
        self.max_pending_items = max_pending_items
        self.max_pending_bytes = max_pending_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ec-mooncake-store",
        )
        self._pending: dict[str, tuple[Future[None], _StoreSave]] = {}
        self._pending_storages: dict[int, tuple[torch.UntypedStorage, int]] = {}
        self._pending_lock = threading.Lock()
        self._pending_bytes = 0
        self._step_candidates: set[str] = set()
        self._closed = False
        self._fatal_error: BaseException | None = None

    def save_output(self, identifier: str, tensor: torch.Tensor) -> None:
        if identifier in self._step_candidates:
            self.reap()
            self._enqueue_save(identifier, tensor)

    def resolve_inputs(
        self,
        expected_tensors: Mapping[str, TensorSpec],
        encoder_cache: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> set[str]:
        """Load outputs before encoding; unresolved items retain normal computation."""
        self.reap()
        self._step_candidates = set(expected_tensors)
        candidates = [key for key in expected_tensors if key not in encoder_cache]
        if not candidates:
            return set()

        pool_keys = [
            make_embedding_key(self.namespace, identifier) for identifier in candidates
        ]
        try:
            exists = self.store_client.batch_exists(pool_keys)
        except (EmbeddingStoreOperationError, OSError):
            logger.warning(
                "Mooncake embedding Store lookup failed; falling back to encoder",
                exc_info=True,
            )
            return set()

        hits = {
            pool_key: expected_tensors[identifier]
            for identifier, pool_key, hit in zip(
                candidates, pool_keys, exists, strict=True
            )
            if hit
        }
        try:
            tensors = self.store_client.load_tensors(hits, device)
        except (EmbeddingStoreOperationError, OSError):
            logger.warning(
                "Mooncake embedding Store GET failed; falling back to encoder",
                exc_info=True,
            )
            return set()
        loaded: set[str] = set()
        for identifier, pool_key in zip(candidates, pool_keys, strict=True):
            if pool_key in tensors:
                encoder_cache[identifier] = tensors[pool_key]
                loaded.add(identifier)
        return loaded

    def _enqueue_save(self, identifier: str, tensor: torch.Tensor) -> bool:
        """Admit contiguous outputs, charging each retained storage once."""
        if not tensor.is_contiguous():
            return False
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        with self._pending_lock:
            if self._fatal_error is not None:
                raise EmbeddingStoreError(
                    "Store publisher has failed"
                ) from self._fatal_error
            if self._closed or identifier in self._pending:
                return False
            _, ref_count = self._pending_storages.get(storage_ptr, (storage, 0))
            budget_bytes = 0 if ref_count else storage.nbytes()
            if (
                len(self._pending) >= self.max_pending_items
                or self._pending_bytes + budget_bytes > self.max_pending_bytes
            ):
                return False
            # Only admitted work creates an event. Runtime errors propagate.
            save = _StoreSave(
                tensor,
                _record_tensor_ready_event(tensor),
                storage_ptr,
            )
            future = self._executor.submit(
                self._save, make_embedding_key(self.namespace, identifier), save
            )
            self._pending[identifier] = (future, save)
            # Keep the storage alive until reaping so its address cannot be reused.
            self._pending_storages[storage_ptr] = (storage, ref_count + 1)
            self._pending_bytes += budget_bytes
        return True

    def _save(self, pool_key: str, save: _StoreSave) -> None:
        try:
            if self.store_client.exists(pool_key):
                return
            tensor = save.tensor
            assert tensor is not None
            if save.ready_event is not None:
                save.ready_event.synchronize()
            self.store_client.put_tensor(pool_key, tensor)
        except BaseException as error:
            # Returned client frames can otherwise keep tensors via a Future's
            # traceback. Uncertain I/O owners remain retained by the client.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            tensor = None
            save.release()

    def reap(self) -> None:
        with self._pending_lock:
            for identifier, (future, save) in list(self._pending.items()):
                # Observe once; a racing completion is handled on the next poll.
                if not future.done():
                    continue
                try:
                    future.result()  # Already done: never waits on Store I/O.
                except (EmbeddingStoreOperationError, OSError) as error:
                    logger.warning("Store PUT failed for %s: %s", identifier, error)
                    # This failure is consumed; do not retain publisher/caller frames.
                    error.__traceback__ = None
                except BaseException as error:
                    # The client may still own unsafe native-I/O buffers. Keep
                    # their charge and fatal result until worker termination.
                    if self._fatal_error is None:
                        self._fatal_error = error
                    continue
                del self._pending[identifier]
                storage, ref_count = self._pending_storages[save.storage_ptr]
                if ref_count == 1:
                    del self._pending_storages[save.storage_ptr]
                    self._pending_bytes -= storage.nbytes()
                else:
                    self._pending_storages[save.storage_ptr] = (storage, ref_count - 1)
            fatal = self._fatal_error
        if fatal is not None:
            raise fatal

    def shutdown(self) -> None:
        with self._pending_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.reap()
        self.store_client.close()


def _record_tensor_ready_event(tensor: torch.Tensor) -> torch.Event | None:
    if tensor.device.type != "cuda":
        return None
    event = torch.Event()
    event.record(torch.accelerator.current_stream(tensor.device))
    return event
