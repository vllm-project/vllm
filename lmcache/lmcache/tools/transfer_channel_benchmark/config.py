# SPDX-License-Identifier: Apache-2.0
"""Configuration and argument parsing for the transfer channel benchmark.

This module is intentionally free of heavy/optional dependencies (torch, the
``lmcache.v1.distributed`` runtime) so the CLI can build its argument parser
without importing them. The runtime imports live in ``benchmark.py``.
"""

# Standard
from dataclasses import dataclass
import argparse

_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}

# Defaults expressed as byte counts so they can be used directly as argparse
# defaults (argparse applies ``type=parse_size`` only to string inputs).
DEFAULT_BUFFER_SIZE = 8 * 1024**3
DEFAULT_PAGE_SIZE = 512 * 1024
DEFAULT_OBJECT_SIZE = 10 * 1024**2


def parse_size(value: str) -> int:
    """Parse a human-friendly size string into a byte count.

    Args:
        value: A size such as ``"8GB"``, ``"512KB"``, ``"10MB"`` or a plain
            integer byte count (e.g. ``"1048576"``).

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    text = str(value).strip().upper()
    for suffix in ("TB", "GB", "MB", "KB", "T", "G", "M", "K", "B"):
        if text.endswith(suffix) and text[: -len(suffix)].strip():
            return int(float(text[: -len(suffix)].strip()) * _UNITS[suffix])
    return int(text)


@dataclass
class BenchmarkConfig:
    """Resolved configuration for one benchmark process (server or client).

    Attributes:
        role: ``"server"`` or ``"client"``.
        transfer_channel_type: The transfer channel implementation to use
            (e.g. ``"nixl"``).
        nixl_backend: The nixl backend name (e.g. ``"UCX"``). Only used by the
            nixl transfer channel type.
        url: Server role binds its transfer-channel server here; client role
            dials this as the peer (server) advertise url to read from.
        listen_url: Client role binds its own (mandatory) transfer-channel
            server here. It never receives reads in this benchmark.
        control_url: Benchmark catalog side-channel. The server binds a ZMQ REP
            socket here; the client connects to fetch the source object catalog.
        buffer_size: Size in bytes of the server's registered L1 region.
        page_size: Page / alignment size in bytes (the L1 ``align_bytes``).
        object_size: Size in bytes of each transferred object (multiple of
            ``page_size``).
        num_objects: Number of objects transferred per read.
        num_source_objects: Number of source objects the server allocates. The
            client reads a random ``num_objects``-sized subset of these.
        use_lazy: Whether the L1 memory manager uses lazy allocation.
        iters: Number of measured read iterations.
        warmup: Number of warmup read iterations (not measured).
        seed: RNG seed for selecting the read subset.
        verify: Whether to verify transferred bytes against a known pattern.
        server_timeout: Seconds the server stays up serving catalog requests.
    """

    role: str
    transfer_channel_type: str = "nixl"
    nixl_backend: str = "UCX"
    url: str = "127.0.0.1:7600"
    listen_url: str = "0.0.0.0:7601"
    control_url: str = "0.0.0.0:7610"
    buffer_size: int = DEFAULT_BUFFER_SIZE
    page_size: int = DEFAULT_PAGE_SIZE
    object_size: int = DEFAULT_OBJECT_SIZE
    num_objects: int = 100
    num_source_objects: int = 0
    use_lazy: bool = False
    iters: int = 5
    warmup: int = 1
    seed: int = 0
    verify: bool = False
    server_timeout: float = 1800.0

    def __post_init__(self) -> None:
        """Apply derived defaults and validate the configuration.

        Raises:
            ValueError: If any field is out of range or mutually inconsistent.
        """
        if self.num_source_objects <= 0:
            self.num_source_objects = 5 * self.num_objects

        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if self.object_size <= 0 or self.object_size % self.page_size != 0:
            raise ValueError(
                f"object_size ({self.object_size}) must be a positive multiple "
                f"of page_size ({self.page_size})"
            )
        if self.num_objects < 1:
            raise ValueError(f"num_objects must be >= 1, got {self.num_objects}")
        if self.num_source_objects < self.num_objects:
            raise ValueError(
                f"num_source_objects ({self.num_source_objects}) must be >= "
                f"num_objects ({self.num_objects})"
            )


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all benchmark arguments to ``parser``.

    Defined here (not in ``benchmark.py``) so the CLI can build its parser
    without importing torch or the distributed runtime.

    Args:
        parser: The argument parser (or subparser) to populate.
    """
    parser.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help="server: register a source buffer and serve its object catalog; "
        "client: read a subset of the server's objects and report throughput.",
    )
    parser.add_argument(
        "--transfer-channel-type",
        default="nixl",
        help="Transfer channel implementation to benchmark.",
    )
    parser.add_argument(
        "--nixl-backend",
        default="UCX",
        help="nixl backend name (nixl-specific), e.g. UCX.",
    )
    parser.add_argument(
        "--url",
        default="127.0.0.1:7600",
        help="server: host:port to bind the transfer-channel server; "
        "client: the server (peer) advertise url to read from.",
    )
    parser.add_argument(
        "--listen-url",
        default="0.0.0.0:7601",
        help="client: host:port for the client's own (mandatory) "
        "transfer-channel server; it never receives reads here.",
    )
    parser.add_argument(
        "--control-url",
        default="0.0.0.0:7610",
        help="benchmark catalog side-channel: server binds here, client "
        "connects here to fetch the source object catalog.",
    )
    parser.add_argument(
        "--buffer-size",
        type=parse_size,
        default=DEFAULT_BUFFER_SIZE,
        help="server: total registered L1 source buffer size (e.g. 8GB).",
    )
    parser.add_argument(
        "--page-size",
        type=parse_size,
        default=DEFAULT_PAGE_SIZE,
        help="page / alignment size; must match on server and client.",
    )
    parser.add_argument(
        "--object-size",
        type=parse_size,
        default=DEFAULT_OBJECT_SIZE,
        help="size of each transferred object (multiple of --page-size).",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=100,
        help="number of objects transferred per read.",
    )
    parser.add_argument(
        "--num-source-objects",
        type=int,
        default=0,
        help="server source pool size; 0 means 5 * --num-objects.",
    )
    parser.add_argument(
        "--use-lazy",
        action="store_true",
        help="use the lazy L1 allocator (experimental for registration).",
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="measured read iterations."
    )
    parser.add_argument("--warmup", type=int, default=1, help="warmup read iterations.")
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for read-subset selection."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify transferred bytes against a known per-object pattern.",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=1800.0,
        help="seconds the server serves catalog requests before exiting.",
    )


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` from parsed arguments.

    Args:
        args: Parsed CLI arguments produced by ``add_benchmark_arguments``.

    Returns:
        The resolved, validated benchmark configuration.
    """
    return BenchmarkConfig(
        role=args.role,
        transfer_channel_type=args.transfer_channel_type,
        nixl_backend=args.nixl_backend,
        url=args.url,
        listen_url=args.listen_url,
        control_url=args.control_url,
        buffer_size=args.buffer_size,
        page_size=args.page_size,
        object_size=args.object_size,
        num_objects=args.num_objects,
        num_source_objects=args.num_source_objects,
        use_lazy=args.use_lazy,
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        verify=args.verify,
        server_timeout=args.server_timeout,
    )
