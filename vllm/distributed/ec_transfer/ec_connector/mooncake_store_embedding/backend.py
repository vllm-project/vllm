# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-side Mooncake Store resolver and publisher."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

from .data import (
    MOONCAKE_TENSOR_METADATA_NBYTES,
    EmbeddingKeyMetadata,
    EmbeddingPoolKey,
    TensorSpec,
)
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
    budget_bytes: int

    def release(self) -> None:
        self.tensor = None
        self.ready_event = None


class MooncakeEmbeddingStoreBackend:
    """Resolve immutable encoder outputs and publish misses asynchronously."""

    def __init__(
        self,
        store_client: MooncakeEmbeddingStoreClient,
        key_metadata: EmbeddingKeyMetadata,
        *,
        max_pending_items: int,
        max_pending_bytes: int,
    ) -> None:
        self.store_client = store_client
        self.key_metadata = key_metadata
        self.max_pending_items = max_pending_items
        self.max_pending_bytes = max_pending_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ec-mooncake-store",
        )
        self._pending: dict[str, tuple[Future[None], _StoreSave]] = {}
        self._pending_lock = threading.Lock()
        self._pending_bytes = 0
        self._step_candidates: set[str] = set()
        self._step_outputs: dict[str, torch.Tensor] = {}
        self._closed = False
        self._fatal_error: BaseException | None = None

    def record_output(self, identifier: str, tensor: torch.Tensor) -> None:
        if identifier in self._step_candidates:
            self._step_outputs.setdefault(identifier, tensor)

    def publish_outputs(self) -> None:
        """Publish outputs recorded in this Encoder step."""
        outputs, self._step_outputs = self._step_outputs, {}
        self._step_candidates.clear()
        for identifier, tensor in outputs.items():
            self._enqueue_save(identifier, tensor)

    def reap(self) -> None:
        for identifier, reason in self._drain_completed().items():
            logger.warning("Store PUT failed for %s: %s", identifier, reason)

    def resolve_inputs(
        self,
        expected_tensors: Mapping[str, TensorSpec],
        encoder_cache: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> set[str]:
        """Load outputs before encoding; unresolved items retain normal computation."""
        self._step_candidates = set(expected_tensors)
        self._step_outputs.clear()
        candidates = [key for key in expected_tensors if key not in encoder_cache]
        if not candidates:
            return set()

        pool_keys = [
            EmbeddingPoolKey(self.key_metadata, identifier) for identifier in candidates
        ]
        try:
            exists = self.store_client.batch_exists(pool_keys)
        except (EmbeddingStoreOperationError, OSError):
            logger.warning(
                "Mooncake embedding Store lookup failed; falling back to encoder",
                exc_info=True,
            )
            return set()

        loaded: set[str] = set()
        for identifier, pool_key, hit in zip(
            candidates, pool_keys, exists, strict=True
        ):
            if not hit:
                continue
            try:
                tensor_meta = self.store_client.get_tensor_meta(pool_key)
                expected = expected_tensors[identifier]
                if tensor_meta != expected:
                    raise ValueError(
                        f"Store tensor {tensor_meta} does not match {expected}"
                    )
                target = torch.empty(
                    expected.shape,
                    dtype=_resolve_torch_dtype(expected.dtype),
                    device=device,
                )
                self.store_client.get_tensor_payload(
                    pool_key,
                    target.data_ptr(),
                    tensor_meta.nbytes,
                    MOONCAKE_TENSOR_METADATA_NBYTES,
                    owner=target,
                )
            except (EmbeddingStoreOperationError, ValueError, OSError):
                logger.warning(
                    "Mooncake embedding Store GET failed for identifier=%s; "
                    "falling back to encoder",
                    identifier,
                    exc_info=True,
                )
                continue
            encoder_cache[identifier] = target
            loaded.add(identifier)
        return loaded

    def _enqueue_save(self, identifier: str, tensor: torch.Tensor) -> bool:
        """Admit contiguous outputs using their full retained storage size."""
        if not tensor.is_contiguous():
            return False
        budget_bytes = tensor.untyped_storage().nbytes()
        with self._pending_lock:
            if self._fatal_error is not None:
                raise EmbeddingStoreError(
                    "Store publisher has failed"
                ) from self._fatal_error
            if self._closed or identifier in self._pending:
                return False
            if (
                len(self._pending) >= self.max_pending_items
                or self._pending_bytes + budget_bytes > self.max_pending_bytes
            ):
                return False
            # Only admitted work creates an event. Runtime errors propagate.
            save = _StoreSave(
                tensor,
                _record_tensor_ready_event(tensor),
                budget_bytes,
            )
            try:
                future = self._executor.submit(
                    self._save, EmbeddingPoolKey(self.key_metadata, identifier), save
                )
            except RuntimeError:
                save.release()
                logger.warning("Failed to enqueue Mooncake Store PUT", exc_info=True)
                return False
            self._pending[identifier] = (future, save)
            self._pending_bytes += budget_bytes
        return True

    def _save(self, pool_key: EmbeddingPoolKey, save: _StoreSave) -> None:
        tensor = None
        try:
            if self.store_client.exists(pool_key):
                return
            tensor = save.tensor
            assert tensor is not None
            _wait_tensor_ready_event(save.ready_event)
            self.store_client.put_tensor(pool_key, tensor)
        except BaseException as error:
            # Returned client frames can otherwise keep tensors via a Future's
            # traceback. Uncertain I/O owners remain retained by the client.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            tensor = None
            save.release()

    def _drain_completed(self) -> dict[str, str]:
        failures: dict[str, str] = {}
        with self._pending_lock:
            for identifier, (future, save) in list(self._pending.items()):
                # Observe once; a racing completion is handled on the next poll.
                if not future.done():
                    continue
                try:
                    future.result()  # Already done: never waits on Store I/O.
                except CancelledError:
                    failures[identifier] = "Store publication cancelled"
                except (EmbeddingStoreOperationError, OSError) as error:
                    failures[identifier] = str(error)
                except BaseException as error:
                    # The client may still own unsafe native-I/O buffers. Keep
                    # their charge and fatal result until worker termination.
                    if self._fatal_error is None:
                        self._fatal_error = error
                    continue
                del self._pending[identifier]
                save.release()  # Also clears owners of cancelled queued work.
                self._pending_bytes -= save.budget_bytes
            fatal = self._fatal_error
        if fatal is not None:
            raise fatal
        return failures

    def shutdown(self) -> None:
        with self._pending_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.reap()
        self.store_client.close()


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "torch.float16":
        return torch.float16
    if dtype == "torch.bfloat16":
        return torch.bfloat16
    if dtype == "torch.float32":
        return torch.float32
    raise ValueError(f"unsupported embedding tensor dtype: {dtype}")


def _record_tensor_ready_event(tensor: torch.Tensor) -> torch.Event | None:
    if tensor.device.type != "cuda":
        return None
    event = torch.Event()
    event.record(torch.accelerator.current_stream(tensor.device))
    return event


def _wait_tensor_ready_event(event: torch.Event | None) -> None:
    if event is not None:
        event.synchronize()
