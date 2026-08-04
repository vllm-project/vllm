# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subcommand implementation.

This module provides argument registration via :func:`add_server_arguments`
and the execution orchestrator :func:`run_server_bench` for the end-to-end
LMCache MP cache-server sanity test.

The command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

Usage examples::

    # GPU mode: real CUDA tensors + IPC
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape (multi-group spec)
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --kvcache-shape-spec '(2,32,1024,8,128):float16:32'

    # Run forever starting from sequence 0
    lmcache bench server --rpc-url tcp://localhost:5555
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import itertools
import math
import mmap
import os
import sys
import time

# First Party
from lmcache import torch_dev

# Heavy imports reused by the orchestrator. ``DTYPE_MAP`` is required
# for the ``--kvcache-shape-spec`` help string at parser-registration
# time. On a slim install these symbols are placeholders; the
# ``_require_full_install`` guard inside the helpers module keeps
# orchestration safe.
from lmcache.cli.commands.bench.server_bench.helpers import (
    _DEFAULT_SHAPE_SPEC,
    DTYPE_MAP,
    _allocate_cpu_shm_kv_cache,
    _allocate_gpu_kv_cache,
    _get_chunk_size,
    _process_request,
    _require_full_install,
    _send_register_kv_cache,
    _send_unregister_kv_cache,
    shm_open_pool_as_mmap,
)

if TYPE_CHECKING:
    # Standard
    from collections.abc import Callable

    # First Party
    from lmcache.cli.commands.base import BaseCommand
    from lmcache.cli.profiling import FlameProfiler
    from lmcache.v1.multiprocess.custom_types import KVCache


# Stash the original (full-install) ImportError so the parser-stub
# branch and the orchestrator branch can both surface it verbatim.
__all__ = (
    "add_server_arguments",
    "run_server_bench",
)


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench server`` arguments to *parser*.

    Requires the full LMCache install (torch, zmq, etc.).
    Callers should check ``_IMPORT_ERROR`` before calling this.

    Args:
        parser: The ``ArgumentParser`` for the server bench subcommand.
    """

    parser.add_argument(
        "--rpc-url",
        default="tcp://localhost:5555",
        help=("ZMQ endpoint of the MP server (default: tcp://localhost:5555)"),
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help=(
            "Run mode (default: gpu). In cpu mode the client allocates "
            "POSIX-SHM-backed KV cache tensors and the server maps the "
            "same physical pages."
        ),
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "engine_driven", "lmcache_driven"],
        default="auto",
        help=(
            "Transport routing for STORE/RETRIEVE (default: auto). "
            "`lmcache_driven` forces the server-driven handle path "
            "(REGISTER_KV_CACHE + STORE/RETRIEVE), which supports "
            "both CUDA IPC and CPU SHM for zero-copy transfers. "
            "`engine_driven` forces the worker-side gather/scatter "
            "data path (REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT + "
            "PREPARE/COMMIT). "
            "`auto` keeps the historical mapping: "
            "gpu->lmcache_driven, cpu->engine_driven."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=512,
        help="Tokens per request (default: 512)",
    )

    # -- KV cache shape --
    kv = parser.add_argument_group("KV cache shape")
    kv.add_argument(
        "--kvcache-shape-spec",
        type=str,
        default=_DEFAULT_SHAPE_SPEC,
        help=(
            "KV shape spec. Describes one or more KV layer groups "
            "separated by ';'. "
            "Grammar: "
            "'(kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]'. "
            "Fields: kv_size=2 for classical K/V or 1 for MLA, "
            "NB=num_blocks, BS=block_size (tokens/block), "
            "NH=num_heads, HS=head_size (elements). "
            "dtype is the element dtype (supported: %s); 'uint8' "
            "is used for FP8-quantized KV. 'layers' is the number "
            "of consecutive layers sharing this group's geometry. "
            "Multi-group example (MLA + classical attention): "
            "'(1,1024,16,1,128):float16:4;"
            "(2,1024,16,8,128):float16:28'. "
            "All groups must share the same NB and BS. "
            "See lmcache.v1.kv_layer_groups.parse_kvcache_shape_spec "
            "for the authoritative parser. Default: '%s'"
            % (", ".join(DTYPE_MAP.keys()), _DEFAULT_SHAPE_SPEC)
        ),
    )
    kv.add_argument(
        "--num-blocks",
        type=int,
        default=1024,
        help="Paged blocks (default: 1024)",
    )
    kv.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per block (default: 16)",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=("Ending sequence number (exclusive). If not set, runs forever."),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help=("Seconds between requests (default: 0.5)"),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help=("HTTP base URL for checksum API (default: http://localhost:8080)"),
    )

    prof = parser.add_argument_group(
        "server profiling",
        "Flame-graph the MP server process while this benchmark drives "
        "load into it. The server's store path (hashing, allocation, "
        "gather, D2H) runs in its own process, not in this client, so "
        "profiling attaches to --profile-server-pid rather than to the "
        "benchmark. See 'lmcache tool flamegraph' for the standalone form.",
    )
    prof.add_argument(
        "--flamegraph",
        choices=["on", "off"],
        default="off",
        help="Record a flame graph of the server during the run (default: off).",
    )
    prof.add_argument(
        "--profile-server-pid",
        type=int,
        default=0,
        metavar="PID",
        help=(
            "Server process to profile, e.g. $(pgrep -f 'lmcache server'). "
            "Required when --flamegraph on."
        ),
    )
    prof.add_argument(
        "--flamegraph-mode",
        default="gil",
        metavar="MODE[,MODE...]",
        help=(
            "What to sample in the server (default: gil). Pass several "
            "comma-separated to profile one load run per mode. Modes: on-cpu, "
            "off-cpu, wakeup, offwake (perf/bcc), wall, gil (py-spy). perf/bcc "
            "name Python functions only when the server was launched with "
            "PYTHONPERFSUPPORT=1. See the 'lmcache tool flamegraph' docs."
        ),
    )
    prof.add_argument(
        "--flamegraph-output",
        default="",
        metavar="PATH",
        help=(
            "SVG output path. Default: "
            "/tmp/lmcache_bench_flames/server-pid<PID>.<mode>.svg."
        ),
    )
    prof.add_argument(
        "--flamegraph-scripts-dir",
        default="",
        metavar="DIR",
        help=(
            "Directory with the FlameGraph scripts (flamegraph.pl, "
            "stackcollapse-perf.pl); default ~/FlameGraph (cloned there on "
            "first use). Unused by --flamegraph-mode wall / gil, which "
            "render their own SVG."
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_server_profiler(
    args: argparse.Namespace,
    log: "Callable[[str], None]",
) -> "FlameProfiler | None":
    """Build a profiler attached to the server, or ``None`` if disabled.

    Validates the target pid and toolchain eagerly so a misconfigured run
    fails before any load is sent. The returned profiler is not started;
    the caller wraps the load loop with ``start`` / ``stop``.

    Args:
        args: Parsed CLI arguments for ``lmcache bench server``.
        log: Progress logger.

    Returns:
        A ready :class:`FlameProfiler` targeting ``--profile-server-pid``,
        or ``None`` when ``--flamegraph`` is off.
    """
    if getattr(args, "flamegraph", "off") != "on":
        return None

    # First Party
    from lmcache.cli.profiling import (
        PY_SPY_MODES,
        FlameProfiler,
        ProfileError,
        check_profiling_deps,
        default_output_path,
        resolve_flamegraph_dir,
    )

    pid = args.profile_server_pid
    if pid <= 0:
        print(
            "Error: --flamegraph on requires --profile-server-pid "
            "(the pid of the running 'lmcache server').",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Error: no such process: --profile-server-pid {pid}", file=sys.stderr)
        sys.exit(2)
    except PermissionError:
        print(
            f"Error: server pid {pid} belongs to another user; "
            "profiling it needs root.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        check_profiling_deps(args.flamegraph_mode)
        flamegraph_dir = ""
        if args.flamegraph_mode not in PY_SPY_MODES:
            flamegraph_dir = resolve_flamegraph_dir(args.flamegraph_scripts_dir, log)
        output = args.flamegraph_output or default_output_path(
            f"server-pid{pid}", args.flamegraph_mode
        )
        return FlameProfiler(
            mode=args.flamegraph_mode,
            output=output,
            flamegraph_dir=flamegraph_dir,
            pid=pid,
            title=f"{args.flamegraph_mode} (server pid {pid})",
        )
    except ProfileError as e:
        print(
            "Error: --flamegraph on was requested but profiling is "
            f"unavailable:\n  {e}",
            file=sys.stderr,
        )
        sys.exit(2)


def run_server_bench(
    command: "BaseCommand",
    args: argparse.Namespace,
) -> None:
    """Centralized orchestrator: run the server bench loop.

    Args:
        command: The owning :class:`BaseCommand` instance, used to
            obtain a configured :class:`Metrics` object via
            ``command.create_metrics``.
        args: Parsed CLI arguments for ``lmcache bench server``.
    """
    _require_full_install()

    # Heavy imports — safe now that _require_full_install passed.
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.kv_layer_groups import (
        format_kvcache_shape_spec,
        parse_kvcache_shape_spec,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.mq import MessageQueueClient

    quiet = getattr(args, "quiet", False)

    def log(msg: str) -> None:
        """Print progress messages; suppressed by --quiet."""
        if not quiet:
            print(msg)

    use_gpu = args.mode == "gpu"
    if use_gpu and not torch_dev.is_available():
        print("ERROR: --mode gpu requires CUDA")
        sys.exit(1)

    # Resolve transfer mode. ``auto`` reproduces the historical
    # behaviour: gpu -> lmcache_driven path, cpu -> engine_driven path.
    # ``lmcache_driven`` / ``engine_driven`` are explicit overrides.
    transfer_mode = getattr(args, "transfer_mode", "auto")
    if transfer_mode == "auto":
        use_handle = use_gpu
    elif transfer_mode == "lmcache_driven":
        use_handle = True
    else:
        use_handle = False
    if use_handle and not use_gpu:
        log(
            "  [info] --transfer-mode=lmcache_driven on cpu mode: "
            "using REGISTER_KV_CACHE + STORE/RETRIEVE over POSIX SHM"
        )

    # The profiler targets the server process (--profile-server-pid), not
    # this benchmark client. Build it before opening any connection so a
    # bad pid or a missing toolchain fails immediately, not after a full
    # benchmark has already run. ``None`` when --flamegraph is off.
    profiler = _build_server_profiler(args, log)

    total_requests = 0
    total_checksum_ok = 0
    total_checksum_fail = 0

    # Latency collectors: keyed by (pass_label, op_type).
    # Each entry is a list of latency values in ms.
    cold_lookup_ms: list[float] = []
    cold_store_ms: list[float] = []
    warm_lookup_ms: list[float] = []
    warm_retrieve_ms: list[float] = []

    url = args.rpc_url
    log(
        "Connecting to LMCache MP Server at %s (mode=%s) ..." % (url, args.mode),
    )

    ctx = zmq.Context()
    client = MessageQueueClient(url, ctx)

    # Tracks whether REGISTER_KV_CACHE succeeded so the ``finally`` block
    # only deregisters a context that was actually registered.
    registered = False

    try:
        # Query chunk size from server
        chunk_size = _get_chunk_size(client)
        log("Server chunk_size = %d" % chunk_size)

        # Parse KV shape spec
        layer_groups = parse_kvcache_shape_spec(args.kvcache_shape_spec)
        # One block-id list is sent per LMCache KV group; each shape-spec
        # group becomes its own group server-side.
        num_engine_group_infos = len(layer_groups) or 1
        # Echo the resolved spec so operators can verify that their
        # input was interpreted as intended. The echoed string is a
        # valid ``--kvcache-shape-spec`` itself.
        log(
            "Resolved KV shape spec: %s" % format_kvcache_shape_spec(layer_groups),
        )
        # Paged KV demands identical ``NB`` / ``BS`` across all groups
        # (block_id -> slot maths is shared), but ``kv_size`` / ``NH`` /
        # ``HS`` / ``dtype`` may vary per group. ``_allocate_gpu_kv_cache(
        # groups=...)`` honours each group's own shape; ``_process_request``
        # only needs a single ``block_size`` / ``total_blocks``.
        first = layer_groups[0]
        nb_vals = {g.shape_desc.nb for g in layer_groups}
        bs_vals = {g.shape_desc.bs for g in layer_groups}
        if len(nb_vals) > 1 or len(bs_vals) > 1:
            raise ValueError(
                "All groups must share NB and BS (paged KV "
                "requires uniform block geometry). Got NB=%s BS=%s"
                % (sorted(nb_vals), sorted(bs_vals))
            )
        num_layers = sum(g.num_layers for g in layer_groups)
        spec_nb = getattr(first.shape_desc, "nb", 0) or 0
        spec_bs = getattr(first.shape_desc, "bs", 0) or 0
        num_blocks = spec_nb if spec_nb > 0 else args.num_blocks
        block_size = spec_bs if spec_bs > 0 else args.block_size
        if spec_nb and spec_nb != args.num_blocks:
            log(
                "  [info] spec nb=%d overrides --num-blocks=%d"
                % (spec_nb, args.num_blocks)
            )
        if spec_bs and spec_bs != args.block_size:
            log(
                "  [info] spec bs=%d overrides --block-size=%d"
                % (spec_bs, args.block_size)
            )
        # For display / legacy hint fields only: collapse to the first
        # group when homogeneous, otherwise report "mixed".
        heads_set = {g.shape_desc.nh for g in layer_groups}
        hs_set = {g.shape_desc.hs for g in layer_groups}
        kv_size_set = {g.shape_desc.kv_size for g in layer_groups}
        dtype_set = {g.dtype for g in layer_groups}
        num_heads_disp: int | str = (
            first.shape_desc.nh if len(heads_set) == 1 else "mixed"
        )
        head_size_disp: int | str = first.shape_desc.hs if len(hs_set) == 1 else "mixed"
        kv_size_disp: int | str = (
            first.shape_desc.kv_size if len(kv_size_set) == 1 else "mixed"
        )
        if len(dtype_set) == 1:
            dtype_str = next(
                (k for k, v in DTYPE_MAP.items() if v == first.dtype),
                "float16",
            )
        else:
            dtype_str = "mixed"

        # Build layout_hints. dtype is sent as a string ("float16")
        # because torch.dtype is not msgpack-serializable. For
        # heterogeneous multi-group specs, per-layer fields (heads /
        # head_size / dtype / kv_size) are reported as "mixed" —
        # ``layout_hints`` is only consumed by the server to pick a
        # ``kv_layout``; the real per-layer shape is discovered from
        # the tensors themselves.
        layout_hints = {
            "num_layers": num_layers,
            "num_heads": num_heads_disp,
            "head_size": head_size_disp,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "dtype": dtype_str,
        }
        # Tell the server each group's true tokens-per-paged-chunk
        # explicitly. Otherwise the server falls back to the block size
        # discovered from the tensors (``shape_desc.bs``), which on the
        # CPU/HND path can be the per-block ``num_heads`` value instead
        # of the real ``block_size`` (HND swaps NH and BS in the tensor
        # shape), and STORE/RETRIEVE would then expect twice as many
        # block IDs as the bench client actually sends.
        engine_group_infos = [
            EngineGroupInfo(
                engine_group_id=group_idx,
                layer_indices=tuple(group.layer_indices),
                tokens_per_block=block_size,
            )
            for group_idx, group in enumerate(layer_groups)
        ]

        num_tokens = args.num_tokens
        log(
            "Each request: %d tokens (%d full chunks)"
            % (
                num_tokens + 1,
                (num_tokens + 1) // chunk_size,
            )
        )
        log(
            "KV shape: %d layers, %s heads x %s, "
            "dtype=%s, blocks=%dx%d, kv=%s"
            % (
                num_layers,
                num_heads_disp,
                head_size_disp,
                dtype_str,
                num_blocks,
                block_size,
                kv_size_disp,
            )
        )

        # Allocate KV tensors. GPU mode wraps real CUDA tensors
        # via CUDA IPC; CPU mode allocates POSIX-SHM-backed
        # tensors so the server can map the same physical pages.
        # shm_names tracks per-layer SHM segment names allocated
        # on demand (one per layer) so we can shm_unlink on exit.
        shm_names: list[str] = []
        if use_gpu:
            # First Party
            from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

            allocated = _allocate_gpu_kv_cache(groups=layer_groups)
            log(
                "Allocated %d GPU tensors on %s" % (len(allocated), allocated[0].device)
            )
            kv_wrappers: KVCache = [CudaIPCWrapper(t) for t in allocated]
            # Keep the CUDA tensors alive for the lifetime of the
            # bench process -- storage may be reclaimed otherwise --
            # and reuse the same list as the client-side data-mode
            # source/sink for the round-trip self-check.
            client_kv_tensors = allocated
        else:
            # First Party
            from lmcache.v1.platform.cpu.shm import CpuShmTensorWrapper

            shm_prefix = CpuShmTensorWrapper.SHM_NAME_PREFIX + str(os.getpid())
            cpu_tensors, cpu_wrappers, shm_names = _allocate_cpu_shm_kv_cache(
                groups=layer_groups, shm_prefix=shm_prefix
            )
            log(
                "Allocated %d CPU SHM tensors (prefix=%s)"
                % (len(cpu_tensors), shm_prefix)
            )
            kv_wrappers = list(cpu_wrappers)
            client_kv_tensors = cpu_tensors

        # Register KV cache before any store/retrieve. In handle mode
        # both GPU (CUDA-IPC) and CPU (POSIX-SHM) paths share the same
        # ``REGISTER_KV_CACHE`` protocol since ``CpuShmTensorWrapper``
        # is a ``DeviceIPCWrapper`` subclass on the wire. In data mode
        # we fall through to the non-GPU registration protocol.
        register_result = _send_register_kv_cache(
            client,
            layout_hints=layout_hints,
            kv_caches=kv_wrappers if use_handle else None,
            use_gpu=use_gpu,
            use_handle=use_handle,
            engine_group_infos=engine_group_infos,
        )
        log("REGISTER_KV_CACHE: %s" % ("OK" if register_result else "FAIL"))
        log("")
        # Mark the registration so the ``finally`` block knows to send the
        # matching UNREGISTER. The data-mode register returns a response
        # object (truthy) and the handle-mode register returns a bool;
        # either way a truthy result means the server holds our context.
        registered = bool(register_result)

        # In data mode the server reply carries the SHM pool name
        # and size; the bench mmaps the same pool so STORE/RETRIEVE
        # can exchange tensor data via slot descriptors instead of
        # round-tripping pickle through the RPC layer.
        server_pool: "mmap.mmap | None" = None
        if not use_handle and not isinstance(register_result, bool):
            shm_name = getattr(register_result, "shm_name", "")
            pool_size = getattr(register_result, "pool_size", 0)
            if shm_name and pool_size > 0:
                server_pool = shm_open_pool_as_mmap(shm_name, pool_size)

        if args.end is not None:
            seq_iter: itertools.count | range = range(args.start, args.end)
        else:
            seq_iter = itertools.count(args.start)

        http_base = args.url.rstrip("/")

        # In data mode the server has no paged ``kv_tensors`` view to
        # hash, so we self-check on the client: cold pass captures
        # ground truth, warm pass zero-fills + re-hashes after
        # RETRIEVE. Handle mode keeps the legacy server-side
        # ``/cache/checksums`` path.
        client_tensors = None if use_handle else client_kv_tensors

        # Record only the steady-state load, not the one-time registration.
        if profiler is not None:
            profiler.start(log)

        for seq_no in seq_iter:
            log("=== Request seq=%d ===" % seq_no)

            # Pass 1: cold (miss -> store)
            cold_result = _process_request(
                client,
                seq_no,
                num_tokens,
                chunk_size,
                "cold",
                http_base=http_base,
                block_size=block_size,
                total_blocks=num_blocks,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                server_pool=server_pool,
            )
            if cold_result is not None:
                if cold_result.lookup_ms is not None:
                    cold_lookup_ms.append(cold_result.lookup_ms)
                if cold_result.store_ms is not None:
                    cold_store_ms.append(cold_result.store_ms)

            time.sleep(args.interval)

            # Pass 2: warm (hit -> retrieve)
            warm_result = _process_request(
                client,
                seq_no,
                num_tokens,
                chunk_size,
                "warm",
                http_base=http_base,
                block_size=block_size,
                total_blocks=num_blocks,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                server_pool=server_pool,
            )
            if warm_result is not None:
                if warm_result.lookup_ms is not None:
                    warm_lookup_ms.append(warm_result.lookup_ms)
                if warm_result.retrieve_ms is not None:
                    warm_retrieve_ms.append(warm_result.retrieve_ms)

            # Compare checksums
            total_requests += 1
            cold_checksums = cold_result.checksums if cold_result else None
            warm_checksums = warm_result.checksums if warm_result else None
            if cold_checksums and warm_checksums:
                if cold_checksums == warm_checksums:
                    total_checksum_ok += 1
                    log("  [seq %d] CHECKSUM MATCH OK" % seq_no)
                else:
                    total_checksum_fail += 1
                    log("  [seq %d] CHECKSUM MISMATCH!" % seq_no)
                    for i, (c, w) in enumerate(
                        zip(
                            cold_checksums,
                            warm_checksums,
                            strict=False,
                        )
                    ):
                        log(
                            "    chunk %d: cold=%s warm=%s %s"
                            % (
                                i,
                                c[:12],
                                w[:12],
                                ("OK" if c == w else "FAIL"),
                            )
                        )

            log("")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("\nStopping...")
    finally:
        # Stop recording once load ends, before teardown
        if profiler is not None:
            profiler.stop(log)
        # Deregister our context from the server before tearing down the
        # client. Otherwise the server keeps the registration (and the
        # CUDA-IPC / POSIX-SHM mappings it holds) alive forever, leaking
        # one context entry per bench run. Must run while the client is
        # still connected, hence before ``client.close()``.
        if registered:
            try:
                ok = _send_unregister_kv_cache(
                    client,
                    instance_id=0,
                    use_handle=use_handle,
                )
                log("UNREGISTER_KV_CACHE: %s" % ("OK" if ok else "FAIL"))
            except zmq.ZMQError as exc:
                log("  [warning] UNREGISTER_KV_CACHE failed: %s" % exc)
        # Release the bench-side mmap of the server SHM pool first
        # (data mode only; ``server_pool`` stays ``None`` otherwise).
        if "server_pool" in locals() and server_pool is not None:
            try:
                server_pool.close()
            except (BufferError, ValueError):
                pass
        client.close()
        ctx.term()
        # Best-effort SHM cleanup so segments don't linger.
        for _name in shm_names if "shm_names" in locals() else []:
            try:
                # First Party
                from lmcache.v1.platform.cpu.shm import shm_unlink

                shm_unlink(_name)
            except OSError:
                pass

    # Emit structured metrics summary.
    _emit_server_bench_metrics(
        command=command,
        args=args,
        total_requests=total_requests,
        total_checksum_ok=total_checksum_ok,
        total_checksum_fail=total_checksum_fail,
        cold_lookup_ms=cold_lookup_ms,
        cold_store_ms=cold_store_ms,
        warm_lookup_ms=warm_lookup_ms,
        warm_retrieve_ms=warm_retrieve_ms,
    )
    log("Done.")


def _emit_server_bench_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    total_requests: int,
    total_checksum_ok: int,
    total_checksum_fail: int,
    cold_lookup_ms: list[float] | None = None,
    cold_store_ms: list[float] | None = None,
    warm_lookup_ms: list[float] | None = None,
    warm_retrieve_ms: list[float] | None = None,
) -> None:
    """Emit server bench summary using the CLI metrics system.

    Args:
        command: The owning :class:`BaseCommand` instance.
        args: Parsed CLI arguments.
        total_requests: Total number of request pairs processed.
        total_checksum_ok: Number of requests with matching checksums.
        total_checksum_fail: Number of requests with mismatched checksums.
        cold_lookup_ms: Per-request cold lookup latencies (ms).
        cold_store_ms: Per-request cold store latencies (ms).
        warm_lookup_ms: Per-request warm lookup latencies (ms).
        warm_retrieve_ms: Per-request warm retrieve latencies (ms).
    """
    if total_requests == 0:
        return

    metrics = command.create_metrics("Server Bench Result", args, width=64)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("rpc_url", "RPC URL", args.rpc_url)
    cfg_section.add("mode", "Mode", args.mode)
    cfg_section.add(
        "transfer_mode", "Transfer mode", getattr(args, "transfer_mode", "auto")
    )
    cfg_section.add("num_tokens", "Tokens / request", args.num_tokens)
    cfg_section.add("interval", "Interval (s)", args.interval)

    result_section = metrics.add_section("results", "Results")
    result_section.add("total_requests", "Total requests", total_requests)
    result_section.add("checksum_ok", "Checksum OK", total_checksum_ok)
    result_section.add("checksum_fail", "Checksum FAIL", total_checksum_fail)
    if total_requests > 0:
        pass_rate = total_checksum_ok / total_requests * 100
        result_section.add("pass_rate", "Pass rate (%)", round(pass_rate, 2))

    # Per-operation latency summary (cold pass).
    _add_latency_section(metrics, "cold_lookup", "Cold Lookup (ms)", cold_lookup_ms)
    _add_latency_section(metrics, "cold_store", "Cold Store (ms)", cold_store_ms)

    # Per-operation latency summary (warm pass).
    _add_latency_section(metrics, "warm_lookup", "Warm Lookup (ms)", warm_lookup_ms)
    _add_latency_section(
        metrics, "warm_retrieve", "Warm Retrieve (ms)", warm_retrieve_ms
    )

    metrics.emit()


def _add_latency_section(
    metrics,
    section_id: str,
    section_title: str,
    latencies: list[float] | None,
) -> None:
    """Add a latency summary section to the metrics report.

    Computes count, mean, min, max, p50, and p99 from the raw
    latency list. Skipped if the list is empty or None.

    Args:
        metrics: The :class:`Metrics` instance.
        section_id: Unique section identifier.
        section_title: Human-readable section title.
        latencies: Raw latency values in milliseconds.
    """
    if not latencies:
        return

    sorted_lat = sorted(latencies)
    count = len(sorted_lat)
    mean = sum(sorted_lat) / count
    p50_idx = max(0, math.ceil(count * 0.50) - 1)
    p99_idx = max(0, math.ceil(count * 0.99) - 1)

    section = metrics.add_section(section_id, section_title)
    section.add(f"{section_id}_count", "count", count)
    section.add(f"{section_id}_mean", "mean", round(mean, 3))
    section.add(f"{section_id}_min", "min", round(sorted_lat[0], 3))
    section.add(f"{section_id}_max", "max", round(sorted_lat[-1], 3))
    section.add(f"{section_id}_p50", "p50", round(sorted_lat[p50_idx], 3))
    section.add(f"{section_id}_p99", "p99", round(sorted_lat[p99_idx], 3))
