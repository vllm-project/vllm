# SPDX-License-Identifier: Apache-2.0
"""Core logic for the transfer channel throughput benchmark.

A *server* process uses an :class:`L1MemoryManager` to initialize a registered
L1 region, allocates a pool of source memory objects in it, registers the
region with a transfer channel, and publishes the source object catalog
(offset/size per object) over a small ZMQ side-channel.

A *client* process fetches the catalog, allocates its own destination objects
via an :class:`L1MemoryManager`, connects the transfer channel, and issues
batched reads of a random subset of the source objects, reporting aggregate
read throughput in GB/s.
"""

# Standard
import argparse
import json
import random
import time

# Third Party
import zmq

# torch is an optional dependency for this benchmark tool. It (and the
# distributed runtime imported below, which also needs torch) is imported here
# rather than in config.py so the `lmcache` CLI can register this tool without
# torch installed. Raise a clear, actionable error if it is missing.
try:
    # Third Party
    import torch
except ImportError as err:
    raise ImportError(
        "PyTorch is required to run the transfer channel benchmark but is not "
        "installed. Install it (e.g. `pip install torch`) and retry."
    ) from err

# First Party
from lmcache.logging import init_logger
from lmcache.tools.transfer_channel_benchmark.config import (
    BenchmarkConfig,
    build_config,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)

logger = init_logger(__name__)

_CATALOG_REQUEST = b"catalog"
_LARGE_LAZY_INIT = 20 * 1024**3
_CLIENT_BUFFER_SLACK = 64 * 1024**2


############################################################
# Helpers
############################################################
def _zmq_endpoint(url: str) -> str:
    """Return a ZMQ tcp endpoint for a ``host:port`` (or pass-through url)."""
    return url if "://" in url else f"tcp://{url}"


def _gbps(num_bytes: int, seconds: float) -> float:
    """Return throughput in GB/s (decimal GB) for ``num_bytes`` over ``seconds``."""
    return (num_bytes / 1e9) / seconds if seconds > 0 else float("inf")


def _create_l1_manager(cfg: BenchmarkConfig, size_bytes: int) -> L1MemoryManager:
    """Create an L1 memory manager sized to ``size_bytes`` for this benchmark."""
    manager_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=cfg.use_lazy,
        init_size_in_bytes=min(_LARGE_LAZY_INIT, size_bytes),
        align_bytes=cfg.page_size,
    )
    return L1MemoryManager(manager_config)


def _object_layout(cfg: BenchmarkConfig) -> MemoryLayoutDesc:
    """Return the layout for a single ``object_size``-byte uint8 object."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([cfg.object_size])], dtypes=[torch.uint8]
    )


############################################################
# Server
############################################################
def server_main(cfg: BenchmarkConfig) -> None:
    """Run the benchmark server: register a buffer and serve its catalog.

    Args:
        cfg: The benchmark configuration (``role == "server"``).

    Raises:
        RuntimeError: If the source pool cannot be allocated in the buffer.
    """
    pool_bytes = cfg.num_source_objects * cfg.object_size
    if pool_bytes > cfg.buffer_size:
        raise RuntimeError(
            f"source pool ({pool_bytes / 1e9:.2f} GB = {cfg.num_source_objects} "
            f"x {cfg.object_size / 1024**2:.1f} MiB) does not fit in the "
            f"{cfg.buffer_size / 1e9:.2f} GB buffer; increase --buffer-size or "
            f"reduce --num-source-objects / --object-size."
        )

    logger.debug(
        "Allocating L1 buffer (%.2f GB) and %d source objects",
        cfg.buffer_size / 1e9,
        cfg.num_source_objects,
    )
    manager = _create_l1_manager(cfg, cfg.buffer_size)
    source_objs: list = []
    try:
        error, source_objs = manager.allocate(
            _object_layout(cfg), cfg.num_source_objects
        )
        if error != L1Error.SUCCESS:
            raise RuntimeError(
                f"Failed to allocate {cfg.num_source_objects} source objects: "
                f"{error}. Increase --buffer-size or reduce "
                f"--num-source-objects / --object-size."
            )

        # Always write a deterministic per-object byte pattern (object index
        # mod 256) so that a client run with --verify can validate the
        # transferred bytes without any extra coordination. This is a one-time
        # cost at startup and does not affect the measured read throughput.
        for index, obj in enumerate(source_objs):
            assert obj.tensor is not None, "Failed to allocate tensor!"
            obj.tensor.fill_(index % 256)

        catalog = [
            (obj.shm_offset, obj.shm_byte_length, index)
            for index, obj in enumerate(source_objs)
        ]

        initialize_transfer_channel_context(
            cfg.transfer_channel_type,
            l1_memory_desc=manager.get_l1_memory_desc(),
            listen_url=cfg.url,
            advertise_url=cfg.url,
            backends=[cfg.nixl_backend],
        )
        try:
            _serve_catalog(cfg, catalog)
        finally:
            delete_transfer_channel_context()
    finally:
        if source_objs:
            manager.free(source_objs)
        manager.close()


def _serve_catalog(cfg: BenchmarkConfig, catalog: list[tuple[int, int, int]]) -> None:
    """Serve the source object catalog over a ZMQ REP socket until timeout."""
    payload = json.dumps(
        {
            "page_size": cfg.page_size,
            "object_size": cfg.object_size,
            "objects": catalog,
        }
    ).encode()

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(_zmq_endpoint(cfg.control_url))
    print(
        f"[server] ready: transfer channel on {cfg.url}, "
        f"catalog on {cfg.control_url} ({len(catalog)} source objects)",
        flush=True,
    )

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline = time.monotonic() + cfg.server_timeout
    try:
        while time.monotonic() < deadline:
            events = dict(poller.poll(timeout=1000))
            if socket in events:
                socket.recv()
                socket.send(payload)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        print("[server] shut down", flush=True)


############################################################
# Client
############################################################
def client_main(cfg: BenchmarkConfig) -> bool:
    """Run the benchmark client: read a subset of source objects and report.

    Args:
        cfg: The benchmark configuration (``role == "client"``).

    Returns:
        ``True`` on success.

    Raises:
        ValueError: If the server's page/object size or pool size is
            incompatible with this client's configuration.
        RuntimeError: If allocation or a read transfer fails.
    """
    catalog = _fetch_catalog(cfg)
    if catalog["page_size"] != cfg.page_size:
        raise ValueError(
            f"page_size mismatch: client {cfg.page_size} vs server "
            f"{catalog['page_size']}; they must match."
        )
    if catalog["object_size"] != cfg.object_size:
        raise ValueError(
            f"object_size mismatch: client {cfg.object_size} vs server "
            f"{catalog['object_size']}; they must match."
        )
    source_objs = catalog["objects"]
    if len(source_objs) < cfg.num_objects:
        raise ValueError(
            f"server has only {len(source_objs)} source objects, fewer than "
            f"--num-objects ({cfg.num_objects})."
        )

    client_buffer = cfg.num_objects * cfg.object_size + _CLIENT_BUFFER_SLACK
    manager = _create_l1_manager(cfg, client_buffer)
    dst_objs: list = []
    try:
        error, dst_objs = manager.allocate(_object_layout(cfg), cfg.num_objects)
        if error != L1Error.SUCCESS:
            raise RuntimeError(
                f"Failed to allocate {cfg.num_objects} destination objects: {error}."
            )

        ctx = initialize_transfer_channel_context(
            cfg.transfer_channel_type,
            l1_memory_desc=manager.get_l1_memory_desc(),
            listen_url=cfg.listen_url,
            advertise_url=cfg.listen_url,
            backends=[cfg.nixl_backend],
        )
        try:
            local_addrs = ctx.get_transfer_channel_address(
                [(obj.shm_offset, obj.shm_byte_length) for obj in dst_objs]
            )
            rng = random.Random(cfg.seed)
            chosen = rng.sample(source_objs, cfg.num_objects)
            remote_addrs = [
                TransferChannelAddress(offset=offset, size=size)
                for offset, size, _ in chosen
            ]
            chosen_indices = [index for _, _, index in chosen]

            connect_start = time.perf_counter()
            client = ctx.get_transfer_channel_client(cfg.url)
            print(
                f"[client] connected in {time.perf_counter() - connect_start:.1f}s; "
                f"reading {cfg.num_objects} objects x "
                f"{cfg.object_size / 1024**2:.1f} MiB",
                flush=True,
            )

            def one_read() -> float:
                start = time.perf_counter()
                task_id = client.submit_read(local_addrs, remote_addrs)
                result = client.query_read_status(task_id)
                while not result.is_finished():
                    result = client.query_read_status(task_id)
                elapsed = time.perf_counter() - start
                succeeded = sum(result.succeeded_mask)
                if succeeded != cfg.num_objects:
                    raise RuntimeError(
                        f"read failed: {succeeded}/{cfg.num_objects} objects succeeded."
                    )
                return elapsed

            for _ in range(cfg.warmup):
                one_read()
            times = [one_read() for _ in range(cfg.iters)]

            if cfg.verify:
                _verify(dst_objs, chosen_indices)

            _report(cfg, times)
            return True
        finally:
            delete_transfer_channel_context()
            del ctx
    finally:
        if dst_objs:
            manager.free(dst_objs)
        manager.close()


def _fetch_catalog(cfg: BenchmarkConfig) -> dict:
    """Fetch the source object catalog from the server's control socket."""
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(cfg.server_timeout * 1000))
    socket.connect(_zmq_endpoint(cfg.control_url))
    try:
        socket.send(_CATALOG_REQUEST)
        try:
            reply = socket.recv()
        except zmq.Again as exc:
            raise RuntimeError(
                f"timed out fetching catalog from {cfg.control_url}; is the "
                f"server running?"
            ) from exc
    finally:
        socket.close(linger=0)
    return json.loads(reply)


def _verify(dst_objs: list, chosen_indices: list[int]) -> None:
    """Check each destination object holds the expected per-object byte pattern.

    Raises:
        RuntimeError: If any destination object does not match.
    """
    for obj, index in zip(dst_objs, chosen_indices, strict=False):
        expected = index % 256
        tensor = obj.tensor
        if tensor is None or not bool((tensor == expected).all()):
            raise RuntimeError(
                f"verify FAILED for source object {index}: expected all bytes "
                f"== {expected}."
            )
    print(
        f"[client] verify OK: {len(dst_objs)} objects match expected pattern",
        flush=True,
    )


def _report(cfg: BenchmarkConfig, times: list[float]) -> None:
    """Print best/median/mean latency and throughput for the measured reads."""
    total_bytes = cfg.num_objects * cfg.object_size
    ordered = sorted(times)
    best = ordered[0]
    median = ordered[len(ordered) // 2]
    mean = sum(ordered) / len(ordered)
    print("\n==== Transfer channel read throughput ====", flush=True)
    print(f"  channel type     : {cfg.transfer_channel_type}", flush=True)
    print(
        f"  payload per read : {total_bytes / 1e9:.3f} GB "
        f"({cfg.num_objects} objs x {cfg.object_size / 1024**2:.1f} MiB)",
        flush=True,
    )
    print(f"  iterations       : {cfg.iters} (warmup {cfg.warmup})", flush=True)
    print(
        f"  best   : {best * 1e3:8.2f} ms   {_gbps(total_bytes, best):7.2f} GB/s",
        flush=True,
    )
    print(
        f"  median : {median * 1e3:8.2f} ms   {_gbps(total_bytes, median):7.2f} GB/s",
        flush=True,
    )
    print(
        f"  mean   : {mean * 1e3:8.2f} ms   {_gbps(total_bytes, mean):7.2f} GB/s",
        flush=True,
    )


############################################################
# Entry
############################################################
def run_benchmark(args: argparse.Namespace) -> bool:
    """Dispatch to the server or client role.

    Args:
        args: Parsed CLI arguments produced by ``add_benchmark_arguments``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    cfg = build_config(args)
    if cfg.role == "server":
        server_main(cfg)
        return True
    return client_main(cfg)
