# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for resource management in hf3fs_client.py: constructor failure cleanup
and idempotent close().  Tests use mock to replace real I/O operations
(hf3fs_fuse.io, SharedMemory, os, CUDA).  
Requires hf3fs_fuse.io to be installed; skipped otherwise.
"""

import pytest
from unittest.mock import MagicMock, patch

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import (  # noqa: F401
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except Exception:
    HF3FS_AVAILABLE = False

requires_hf3fs = pytest.mark.skipif(
    not HF3FS_AVAILABLE,
    reason="hf3fs_fuse.io is not available on this machine",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeShm:
    """Shared-memory stub matching the multiprocessing.shared_memory.SharedMemory
    interface used by Hf3fsClient:

    Attributes accessed by the constructor:
      .buf      – memoryview / buffer-protocol object consumed by torch.frombuffer
    Methods called during normal lifetime:
      .unlink() – called right after the iovec is set up
      .close()  – called in _release_resources()
    """

    def __init__(self, size: int = 1024):
        self._data = bytearray(size)
        self.buf = memoryview(self._data)
        self.closed = False
        self.close_call_count = 0
        self.unlink_call_count = 0

    def close(self):
        self.closed = True
        self.close_call_count += 1

    def unlink(self):
        self.unlink_call_count += 1


# ===========================================================================
# TestHf3fsClientResourceManagement
# ===========================================================================


@requires_hf3fs
class TestHf3fsClientResourceManagement:
    """Tests for constructor failure cleanup and idempotent close()."""

    _MOD = "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client"

    # ------------------------------------------------------------------
    # Helper: build a minimal Hf3fsClient bypassing all real I/O so that
    # we can fully control its internal state.
    # ------------------------------------------------------------------

    def _make_client(self, tmp_path):
        """Return a fully-mocked Hf3fsClient with controllable internals."""
        fake_shm_r = _FakeShm()
        fake_shm_w = _FakeShm()

        patcher_list = [
            patch(f"{self._MOD}.HF3FS_AVAILABLE", True),
            patch(f"{self._MOD}.register_fd"),
            patch(f"{self._MOD}.deregister_fd"),
            patch(f"{self._MOD}.extract_mount_point", return_value="/mnt/hf3fs"),
            patch(f"{self._MOD}.make_ioring", return_value=MagicMock()),
            patch(f"{self._MOD}.make_iovec", return_value=MagicMock()),
            patch(
                "multiprocessing.shared_memory.SharedMemory",
                side_effect=[fake_shm_r, fake_shm_w],
            ),
            patch("os.open", return_value=99),
            patch("os.ftruncate"),
            patch("os.close"),
            patch("os.fsync"),
            patch("torch.cuda.Stream", return_value=MagicMock()),
            patch("torch.frombuffer", return_value=MagicMock()),
            patch("torch.empty", return_value=MagicMock()),
        ]
        for p in patcher_list:
            p.start()

        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client import (
                Hf3fsClient,
            )
            client = Hf3fsClient(
                path=str(tmp_path / "test.bin"),
                size=1024,
                bytes_per_page=256,
                entries=4,
            )
        finally:
            for p in patcher_list:
                p.stop()

        # Manually point internal handles to our controllable fakes so that
        # assertions after close() can inspect them directly.
        client.shm_r = fake_shm_r
        client.shm_w = fake_shm_w
        client.file = 99
        return client, fake_shm_r, fake_shm_w

    # ------------------------------------------------------------------
    # close() idempotency
    # ------------------------------------------------------------------

    def test_close_idempotent_and_handles_cleared(self, tmp_path):
        """Multiple close() calls must not raise; deregister_fd called exactly
        once, all handles set to None, shm.close() invoked."""
        client, shm_r, shm_w = self._make_client(tmp_path)

        with (
            patch(f"{self._MOD}.deregister_fd") as mock_dereg,
            patch("os.close"),
        ):
            client.close()  # first close
            client.close()  # second close — must be no-op
            client.close()  # third close — must be no-op

        assert client._closed is True
        assert mock_dereg.call_count == 1, (
            f"deregister_fd called {mock_dereg.call_count} times; expected 1"
        )
        for attr in ("iov_r", "iov_w", "ior_r", "ior_w", "shm_r", "shm_w", "file"):
            assert getattr(client, attr) is None, f"{attr} should be None after close()"
        assert shm_r.closed is True
        assert shm_w.closed is True

    def test_flush_after_close_is_noop(self, tmp_path):
        """flush() after close() must silently do nothing (no fsync call)."""
        client, _, _ = self._make_client(tmp_path)

        with (
            patch(f"{self._MOD}.deregister_fd"),
            patch("os.close"),
            patch("os.fsync") as mock_fsync,
        ):
            client.close()
            client.flush()

        mock_fsync.assert_not_called()

    # ------------------------------------------------------------------
    # Constructor failure leaves no leaked resources
    # ------------------------------------------------------------------

    def test_constructor_failure_after_file_open_cleans_file(self, tmp_path):
        """If the constructor raises after os.open(), the fd must be closed."""
        with (
            patch(f"{self._MOD}.HF3FS_AVAILABLE", True),
            patch(f"{self._MOD}.register_fd"),
            patch(f"{self._MOD}.deregister_fd"),
            patch(
                f"{self._MOD}.extract_mount_point",
                side_effect=RuntimeError("mount point not found"),
            ),
            patch("os.open", return_value=55),
            patch("os.ftruncate"),
            patch("os.close") as mock_os_close,
            patch("torch.cuda.Stream", return_value=MagicMock()),
        ):
            from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client import (
                Hf3fsClient,
            )
            with pytest.raises(RuntimeError, match="mount point not found"):
                Hf3fsClient(
                    path=str(tmp_path / "fail.bin"),
                    size=1024,
                    bytes_per_page=256,
                    entries=4,
                )

        mock_os_close.assert_called_once_with(55)

    def test_constructor_failure_after_shm_alloc_closes_shm(self, tmp_path):
        """Constructor raises after SharedMemory creation → both shm objects closed."""
        fake_shm_r = _FakeShm()
        fake_shm_w = _FakeShm()

        with (
            patch(f"{self._MOD}.HF3FS_AVAILABLE", True),
            patch(f"{self._MOD}.register_fd"),
            patch(f"{self._MOD}.deregister_fd"),
            patch(f"{self._MOD}.extract_mount_point", return_value="/mnt/hf3fs"),
            patch(
                "multiprocessing.shared_memory.SharedMemory",
                side_effect=[fake_shm_r, fake_shm_w],
            ),
            patch("os.open", return_value=66),
            patch("os.ftruncate"),
            patch("os.close"),
            patch("torch.frombuffer", return_value=MagicMock()),
            patch("torch.empty", return_value=MagicMock()),
            patch(
                f"{self._MOD}.make_ioring",
                side_effect=RuntimeError("ioring init failed"),
            ),
            patch(f"{self._MOD}.make_iovec", return_value=MagicMock()),
            patch("torch.cuda.Stream", return_value=MagicMock()),
        ):
            from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client import (
                Hf3fsClient,
            )
            with pytest.raises(RuntimeError, match="ioring init failed"):
                Hf3fsClient(
                    path=str(tmp_path / "fail2.bin"),
                    size=1024,
                    bytes_per_page=256,
                    entries=4,
                )

        assert fake_shm_r.closed is True, "shm_r was not closed after constructor failure"
        assert fake_shm_w.closed is True, "shm_w was not closed after constructor failure"

    def test_constructor_failure_does_not_close_unallocated_shm(self, tmp_path):
        """Failure before SharedMemory is created must not raise AttributeError
        or TypeError from cleanup."""
        with (
            patch(f"{self._MOD}.HF3FS_AVAILABLE", True),
            patch(f"{self._MOD}.register_fd"),
            patch(f"{self._MOD}.deregister_fd"),
            patch(
                f"{self._MOD}.extract_mount_point",
                side_effect=RuntimeError("early failure"),
            ),
            patch("os.open", return_value=77),
            patch("os.ftruncate"),
            patch("os.close"),
            patch("torch.cuda.Stream", return_value=MagicMock()),
        ):
            from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client import (
                Hf3fsClient,
            )
            with pytest.raises(RuntimeError, match="early failure"):
                Hf3fsClient(
                    path=str(tmp_path / "early_fail.bin"),
                    size=1024,
                    bytes_per_page=256,
                    entries=4,
                )

    # ------------------------------------------------------------------
    # _release_resources on already-cleared state must be a no-op
    # ------------------------------------------------------------------

    def test_release_resources_on_empty_state_is_safe(self, tmp_path):
        """_release_resources() on a fully-cleared client must not raise."""
        client, _, _ = self._make_client(tmp_path)

        with (
            patch(f"{self._MOD}.deregister_fd"),
            patch("os.close"),
        ):
            client.close()  # clears all handles

        with (
            patch(f"{self._MOD}.deregister_fd") as mock_dereg2,
            patch("os.close") as mock_os_close2,
        ):
            client._release_resources()  # must not raise

        mock_dereg2.assert_not_called()
        mock_os_close2.assert_not_called()
