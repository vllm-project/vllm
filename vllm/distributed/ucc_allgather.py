# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UCC Allgather wrapper for async CPU-side collectives over InfiniBand.

This module provides an async allgather implementation using UCC (Unified
Collective Communications) that can overlap with GPU computation. It is
designed for DP rank synchronization where the payload is small (~40-280 bytes)
and can benefit from running over InfiniBand while the GPU forward pass executes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass


class UCCHandle:
    """Handle for non-blocking UCC collective operation.

    This handle tracks the state of an async allgather and provides
    methods to poll for completion or block until complete.
    """

    def __init__(self, handle_ptr: int, ctx: "UCCAllgather"):
        self._handle_ptr = handle_ptr
        self._ctx = ctx
        self._completed = False

    def test(self) -> bool:
        """Non-blocking poll. Returns True if complete.

        This also drives progress on the UCC context to ensure
        the operation can make forward progress.
        """
        if self._completed:
            return True

        # Import here to avoid import errors when UCC is not available
        from vllm import ucc_allgather as mod

        # Drive UCC progress
        self._ctx.progress()

        result = mod.test(self._handle_ptr)
        if result:
            self._completed = True
        return result

    def wait(self) -> None:
        """Block until the collective operation completes."""
        if self._completed:
            return

        from vllm import ucc_allgather as mod

        mod.wait(self._handle_ptr)
        self._completed = True

    def __del__(self):
        """Clean up the handle when garbage collected."""
        if hasattr(self, "_handle_ptr") and self._handle_ptr:
            try:
                from vllm import ucc_allgather as mod

                mod.free_handle(self._handle_ptr)
            except Exception:
                pass  # Ignore errors during cleanup


class UCCAllgather:
    """Async CPU-side allgather using UCC over InfiniBand.

    This class provides an async allgather implementation that can overlap
    with GPU computation. It uses UCC for the collective operation and
    requires an out-of-band (OOB) allgather function for bootstrapping
    the UCC team.

    Example usage:
        def oob_allgather(data: bytes) -> list[bytes]:
            # Use gloo CPU group for OOB
            tensor = torch.tensor(list(data), dtype=torch.uint8)
            output = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(output, tensor, group=cpu_group)
            return [bytes(t.tolist()) for t in output]

        ucc = UCCAllgather(rank, world_size, oob_allgather)

        # Start async operation
        send_buf = struct.pack('5i', *values)
        recv_buf = bytearray(len(send_buf) * world_size)
        handle = ucc.allgather_async(send_buf, recv_buf)

        # Do GPU work here...

        # Wait for completion
        handle.wait()

        # Results are in recv_buf
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        oob_allgather_fn: Callable[[bytes], list[bytes]],
    ):
        """Initialize UCC allgather.

        Args:
            rank: Local rank in the communicator
            world_size: Total number of ranks
            oob_allgather_fn: Out-of-band allgather function for bootstrap.
                This function takes bytes and returns a list of bytes from
                all ranks. It is used during UCC team creation.
        """
        from vllm import ucc_allgather as mod

        self._rank = rank
        self._world_size = world_size
        self._ctx_ptr = mod.init(rank, world_size)
        self._oob_allgather_fn = oob_allgather_fn
        self._team_created = False

    def _ensure_team(self) -> None:
        """Create the UCC team if not already created."""
        if self._team_created:
            return

        from vllm import ucc_allgather as mod

        mod.create_team(self._ctx_ptr, self._oob_allgather_fn)
        self._team_created = True

    def allgather_async(
        self, send_buf: bytes | memoryview, recv_buf: bytearray
    ) -> UCCHandle:
        """Start an async allgather operation.

        Args:
            send_buf: Data to send from this rank. Can be bytes or memoryview.
            recv_buf: Buffer to receive data from all ranks. Must be a
                bytearray with size = len(send_buf) * world_size.

        Returns:
            UCCHandle that can be used to poll or wait for completion.
        """
        self._ensure_team()

        from vllm import ucc_allgather as mod

        handle_ptr = mod.allgather_async(self._ctx_ptr, send_buf, recv_buf)
        return UCCHandle(handle_ptr, self)

    def progress(self) -> None:
        """Drive progress on the UCC context.

        This should be called periodically to ensure async operations
        can make forward progress, especially when using test() to poll.
        """
        from vllm import ucc_allgather as mod

        mod.progress(self._ctx_ptr)

    def destroy(self) -> None:
        """Destroy the UCC context and free resources."""
        if self._ctx_ptr:
            from vllm import ucc_allgather as mod

            mod.destroy(self._ctx_ptr)
            self._ctx_ptr = None
            self._team_created = False

    def __del__(self):
        """Clean up when garbage collected."""
        try:
            self.destroy()
        except Exception:
            pass  # Ignore errors during cleanup


# Module-level singleton for the UCC allgather instance
_ucc_allgather: UCCAllgather | None = None
_ucc_available: bool | None = None


def is_ucc_available() -> bool:
    """Check if UCC extension is available."""
    global _ucc_available
    if _ucc_available is None:
        try:
            from vllm import ucc_allgather  # noqa: F401

            _ucc_available = True
        except ImportError:
            _ucc_available = False
    return _ucc_available


def get_ucc_allgather() -> UCCAllgather | None:
    """Get the singleton UCC allgather instance."""
    global _ucc_allgather
    return _ucc_allgather


def init_ucc_allgather(
    rank: int,
    world_size: int,
    oob_allgather_fn: Callable[[bytes], list[bytes]],
) -> UCCAllgather | None:
    """Initialize the singleton UCC allgather instance.

    Args:
        rank: Local rank in the communicator
        world_size: Total number of ranks
        oob_allgather_fn: Out-of-band allgather function for bootstrap

    Returns:
        UCCAllgather instance if UCC is available, None otherwise
    """
    global _ucc_allgather

    if not is_ucc_available():
        return None

    if _ucc_allgather is not None:
        return _ucc_allgather

    _ucc_allgather = UCCAllgather(rank, world_size, oob_allgather_fn)
    return _ucc_allgather


def destroy_ucc_allgather() -> None:
    """Destroy the singleton UCC allgather instance."""
    global _ucc_allgather
    if _ucc_allgather is not None:
        _ucc_allgather.destroy()
        _ucc_allgather = None
