# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own an EGM Store pool until SIGINT or SIGTERM.

Stop all requesters before stopping this provider. See
docs/features/mooncake_store_connector_usage.md for prerequisites and deployment.
"""

import argparse
import signal
import threading


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostname", required=True, help="Reachable provider address")
    parser.add_argument("--master", required=True, help="Mooncake master host:port")
    parser.add_argument(
        "--size-gib",
        type=int,
        default=64,
        help="EGM pool capacity in GiB (default: 64)",
    )
    parser.add_argument(
        "--numa-nodes", default="auto", help="EGM NUMA nodes (default: auto)"
    )
    args = parser.parse_args()
    if args.size_gib <= 0:
        parser.error("--size-gib must be positive")

    from mooncake.store import MooncakeDistributedStore

    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())

    store = MooncakeDistributedStore()
    try:
        result = store.setup(
            {
                "local_hostname": args.hostname,
                "metadata_server": "P2PHANDSHAKE",
                "master_server_addr": args.master,
                "global_segment_size": str(args.size_gib * 1024**3),
                "local_buffer_size": "0",
                "protocol": "nvlink",
                "rdma_devices": "",
                "enable_egm_store_pool": "true",
                "egm_numa_nodes": args.numa_nodes,
            }
        )
        if result != 0:
            raise RuntimeError(f"Mooncake EGM provider setup failed: {result}")
        if not stopped.is_set():
            print(
                f"EGM provider ready: {store.get_hostname()}, "
                f"requested {args.size_gib} GiB. "
                "Stop requesters before sending SIGINT or SIGTERM.",
                flush=True,
            )
        stopped.wait()
    finally:
        result = store.close()
        if result != 0:
            raise RuntimeError(f"Mooncake EGM provider close failed: {result}")


if __name__ == "__main__":
    main()
