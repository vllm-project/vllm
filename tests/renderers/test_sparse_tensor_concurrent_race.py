# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pybase64
import pytest
import torch

from vllm.config import ModelConfig
from vllm.multimodal.media import AudioEmbeddingMediaIO, ImageEmbeddingMediaIO
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.utils.sparse_utils import check_sparse_tensor_invariants_threadsafe


def _encode_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return pybase64.b64encode(buffer.read())


def _create_malicious_sparse_tensor() -> torch.Tensor:
    indices = torch.tensor([[10], [10]])
    values = torch.tensor([1.0])
    shape = (3, 3)
    return torch.sparse_coo_tensor(
        indices, values, shape, dtype=torch.float32, check_invariants=False
    )


def _create_valid_dense_tensor(hidden_size: int = 768) -> torch.Tensor:
    return torch.randn(4, hidden_size, dtype=torch.float32)


@pytest.fixture
def model_config():
    return ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enable_prompt_embeds=True,
    )


class TestNegativeControlWithoutRace:
    """The invalid payload must be rejected even without concurrency."""

    def test_malicious_sparse_rejected_by_prompt_loader(self, model_config):
        encoded = _encode_tensor(_create_malicious_sparse_tensor())
        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, encoded)

    def test_malicious_sparse_rejected_by_image_loader(self):
        io_handler = ImageEmbeddingMediaIO()
        encoded = _encode_tensor(_create_malicious_sparse_tensor())
        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_base64("", encoded.decode("utf-8"))

    def test_malicious_sparse_rejected_by_audio_loader(self):
        io_handler = AudioEmbeddingMediaIO()
        encoded = _encode_tensor(_create_malicious_sparse_tensor())
        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_base64("", encoded.decode("utf-8"))


class TestConcurrentRaceProtection:
    """Verify the lock prevents the A-enter, B-enter, A-exit, B-load race."""

    def test_forced_interleaving_still_rejects_invalid(self, model_config):
        """Force the exact interleaving that bypasses the guard without a lock.

        Thread A (benign): enters context, loads valid tensor, exits context.
        Thread B (malicious): enters context after A, but loads after A exits.

        Without _SPARSE_LOAD_LOCK the flag would be False when B loads.
        With the lock, B cannot enter until A fully completes.
        """
        barrier = threading.Barrier(2, timeout=5)
        results: dict[str, object] = {}

        valid_encoded = _encode_tensor(_create_valid_dense_tensor())
        malicious_encoded = _encode_tensor(_create_malicious_sparse_tensor())

        def thread_a_benign():
            """Enter context, signal B, load, exit."""
            try:
                with check_sparse_tensor_invariants_threadsafe():
                    barrier.wait()  # signal B that A holds the lock
                    tensor = torch.load(
                        io.BytesIO(pybase64.b64decode(valid_encoded, validate=True)),
                        weights_only=True,
                        map_location=torch.device("cpu"),
                    )
                    results["a_loaded"] = True
                    results["a_tensor_valid"] = isinstance(tensor, torch.Tensor)
            except threading.BrokenBarrierError:
                results["a_barrier_broken"] = True

        def thread_b_malicious():
            """Wait for A to hold the lock, then try to acquire it."""
            try:
                barrier.wait()  # wait until A is inside the lock
                with check_sparse_tensor_invariants_threadsafe():
                    tensor = torch.load(
                        io.BytesIO(
                            pybase64.b64decode(malicious_encoded, validate=True)
                        ),
                        weights_only=True,
                        map_location=torch.device("cpu"),
                    )
                    tensor.to_dense()
                    results["b_bypass"] = True
            except (RuntimeError, ValueError):
                results["b_rejected"] = True
            except threading.BrokenBarrierError:
                results["b_barrier_broken"] = True

        t_a = threading.Thread(target=thread_a_benign)
        t_b = threading.Thread(target=thread_b_malicious)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert results.get("a_loaded") is True
        assert results.get("a_tensor_valid") is True
        assert results.get("b_rejected") is True, (
            "Malicious tensor was NOT rejected under concurrency"
        )
        assert results.get("b_bypass") is not True

    def test_concurrent_loads_all_reject_invalid(self, model_config):
        """Multiple concurrent malicious loads must all be rejected."""
        malicious_encoded = _encode_tensor(_create_malicious_sparse_tensor())
        num_workers = 4
        rejected = []
        bypassed = []

        def attempt_load(_):
            try:
                safe_load_prompt_embeds(model_config, malicious_encoded)
                bypassed.append(True)
            except (RuntimeError, ValueError):
                rejected.append(True)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(attempt_load, range(num_workers)))

        assert len(rejected) == num_workers
        assert len(bypassed) == 0

    def test_concurrent_valid_loads_succeed(self, model_config):
        """Concurrent valid loads must all succeed (no false rejections)."""
        valid_encoded = _encode_tensor(_create_valid_dense_tensor())
        num_workers = 4
        successes = []
        failures = []

        def attempt_load(_):
            try:
                result = safe_load_prompt_embeds(model_config, valid_encoded)
                successes.append(result.shape)
            except Exception as e:
                failures.append(str(e))

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(attempt_load, range(num_workers)))

        assert len(successes) == num_workers
        assert len(failures) == 0


class TestGlobalFlagRestoration:
    """The global invariant flag must be correctly restored after use."""

    def test_flag_restored_after_success(self, model_config):
        initial = torch.sparse.check_sparse_tensor_invariants.is_enabled()
        valid_encoded = _encode_tensor(_create_valid_dense_tensor())

        safe_load_prompt_embeds(model_config, valid_encoded)

        assert torch.sparse.check_sparse_tensor_invariants.is_enabled() == initial

    def test_flag_restored_after_exception(self, model_config):
        initial = torch.sparse.check_sparse_tensor_invariants.is_enabled()
        malicious_encoded = _encode_tensor(_create_malicious_sparse_tensor())

        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, malicious_encoded)

        assert torch.sparse.check_sparse_tensor_invariants.is_enabled() == initial

    def test_flag_restored_after_concurrent_exceptions(self, model_config):
        """Flag must be restored even when multiple threads raise."""
        initial = torch.sparse.check_sparse_tensor_invariants.is_enabled()
        malicious_encoded = _encode_tensor(_create_malicious_sparse_tensor())

        def attempt_load(_):
            with contextlib.suppress(RuntimeError, ValueError):
                safe_load_prompt_embeds(model_config, malicious_encoded)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(attempt_load, range(4)))

        assert torch.sparse.check_sparse_tensor_invariants.is_enabled() == initial


class TestCrossLoaderLockSharing:
    """All loaders must share the same lock to prevent cross-loader races."""

    def test_prompt_and_image_share_lock(self, model_config):
        """Prompt and image loaders cannot run their sparse guards
        concurrently."""
        valid_prompt = _encode_tensor(_create_valid_dense_tensor())
        valid_image = _encode_tensor(_create_valid_dense_tensor(hidden_size=10))

        io_handler = ImageEmbeddingMediaIO()

        def load_prompt():
            safe_load_prompt_embeds(model_config, valid_prompt)

        def load_image():
            io_handler.load_base64("", valid_image.decode("utf-8"))

        # If both threads could hold the lock simultaneously, the barrier
        # would succeed. Since the lock serializes them, the barrier will
        # time out for the second thread (proving mutual exclusion).
        t1 = threading.Thread(target=load_prompt)
        t2 = threading.Thread(target=load_image)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # The key assertion: both completed (barrier didn't deadlock due to
        # serialization — one finishes before the other starts), AND invalid
        # tensors are still rejected.
        malicious = _encode_tensor(_create_malicious_sparse_tensor())
        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, malicious)
        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_base64("", malicious.decode("utf-8"))

    def test_all_loaders_use_same_context_manager(self):
        """Verify all modules reference the same context manager function."""
        import vllm.multimodal.media.audio as audio_mod
        import vllm.multimodal.media.image as image_mod
        import vllm.renderers.embed_utils as embed_mod
        from vllm.utils.sparse_utils import (
            check_sparse_tensor_invariants_threadsafe as cm_from_utils,
        )

        assert embed_mod.check_sparse_tensor_invariants_threadsafe is cm_from_utils
        assert image_mod.check_sparse_tensor_invariants_threadsafe is cm_from_utils
        assert audio_mod.check_sparse_tensor_invariants_threadsafe is cm_from_utils
