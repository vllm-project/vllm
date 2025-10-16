import argparse
import os
from typing import Union


def generate_head_script(
    address: str, ray_port: int, num_cpus: str, num_gpus: str, nnodes: int
) -> str:
    num_cpus_str = str(num_cpus) if num_cpus.isdigit() else num_cpus
    num_gpus_str = str(num_gpus) if num_gpus.isdigit() else num_gpus

    script = f"""# Spawning Ray cluster (head node)
echo "Ray: Starting HEAD at $(hostname)..."
export RAY_memory_monitor_refresh_ms=0
ray start \\
    --head \\
    --node-ip-address={address} \\
    --port={ray_port} \\
    --num-cpus {num_cpus_str} \\
    --num-gpus {num_gpus_str}

# Ray cluster needs to be initialized before spawning workers
echo "Waiting for {nnodes} worker nodes to connect..."
START_TIME=$(date +%s)
TIMEOUT=120    # seconds
INTERVAL=1
while :; do
    # Count alive nodes
    WORKER_COUNT=$(python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))')
    if [ "$WORKER_COUNT" -ge "{nnodes}" ]; then
        echo "Ray: ✅ Found all ($WORKER_COUNT) nodes."
        break
    fi
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TIME ))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Ray: ❌ Timeout after $TIMEOUT seconds: not enough workers joined."
        exit 1
    fi
    echo "⏳ Still waiting... ($WORKER_COUNT found)"
    sleep "$INTERVAL"
done
"""
    return script


def get_start_ray_worker_cmd(
    main_address: str,
    ray_port: int,
    num_cpus: Union[int, str],
    num_gpus: Union[int, str],
) -> str:
    num_cpus_str = str(num_cpus) if isinstance(num_cpus, int) else num_cpus
    num_gpus_str = str(num_gpus) if isinstance(num_gpus, int) else num_gpus

    return f"""echo "Ray: Starting WORKER at $(hostname)..."
export RAY_memory_monitor_refresh_ms=0
ray start \\
    --address {main_address}:{ray_port} \\
    --num-cpus {num_cpus_str} \\
    --num-gpus {num_gpus_str} \\
    --block
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Ray cluster scripts for head or worker nodes."
    )

    parser.add_argument(
        "--is_head",
        action="store_true",
        help="Generate script for the head node (default: worker).",
    )
    parser.add_argument(
        "--address", type=str, required=True, help="IP address of the head node."
    )
    parser.add_argument(
        "--ray_port", type=int, required=True, help="Port for the Ray cluster."
    )
    parser.add_argument(
        "--num_cpus",
        type=str,
        required=True,
        help="Number of CPUs to allocate (e.g., '4' or 'auto').",
    )
    parser.add_argument(
        "--num_gpus",
        type=str,
        required=True,
        help="Number of GPUs to allocate (e.g., '1' or 'auto').",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        help="Total number of nodes to wait for (only for head node).",
    )

    args = parser.parse_args()
    print(f"Ray: Args: {args}")

    if args.is_head:
        script = generate_head_script(
            address=args.address,
            ray_port=args.ray_port,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            nnodes=args.nnodes,
        )
    else:
        assert args.nnodes is None, "nnodes is not used for worker nodes"
        script = get_start_ray_worker_cmd(
            main_address=args.address,
            ray_port=args.ray_port,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
        )

    os.system(script)


if __name__ == "__main__":
    main()
