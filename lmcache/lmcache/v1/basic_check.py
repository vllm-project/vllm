# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import asyncio

# First Party
from lmcache.v1.check import registry

model_name = "/lmcache_test_model/"


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache basic check Tool")
    parser.add_argument(
        "--mode",
        required=True,
        help="Operation mode (e.g. test_remote, test_storage_manager). "
        "Use 'list' to show available modes",
    )
    parser.add_argument("--model", default=model_name, help="model name")
    parser.add_argument(
        "--num-keys",
        type=int,
        default=5,
        help="Number of keys for gen mode or test iterations "
        "for test_* modes (default: 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency level for generation (gen mode only)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for key generation (gen mode only)",
    )
    parser.add_argument(
        "--l2-adapter",
        dest="l2_adapter",
        action="append",
        default=[],
        type=str,
        metavar="JSON",
        help="L2 adapter spec as JSON (test_l2_adapter mode). "
        'e.g. \'{"type":"mock","max_size_gb":1}\'.',
    )
    parser.add_argument(
        "--obj-size",
        dest="obj_size",
        type=int,
        default=None,
        help="Object size in number of elements (default: 1024)",
    )
    parser.add_argument(
        "--kv-dtype",
        dest="kv_dtype",
        type=str,
        default=None,
        help="KV dtype, e.g. float32, bfloat16, float16 (default depends on mode)",
    )
    parser.add_argument(
        "--settle-time",
        dest="settle_time",
        type=float,
        default=0.0,
        help="Seconds to wait after store before load "
        "(default: 0, useful for remote backends)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # List available modes if requested
    if args.mode == "list":
        registry.load_modes()
        print("Available check modes:")
        for mode_name in registry.modes:
            print(f"  - {mode_name}")
        return

    # Get the requested mode function
    mode_func = registry.get_mode(args.mode)
    if not mode_func:
        print(
            f"Error: Unknown mode '{args.mode}'. "
            "Use '--mode list' to see available modes."
        )
        return

    # Prepare arguments for the mode function
    mode_args = {
        "model": args.model,
        "num_keys": args.num_keys,
        "concurrency": args.concurrency,
        "offset": args.offset,
        "l2_adapter": args.l2_adapter,
        "obj_size": args.obj_size,
        "kv_dtype": args.kv_dtype,
        "settle_time": args.settle_time,
    }

    # Execute the mode function
    await mode_func(**mode_args)


if __name__ == "__main__":
    asyncio.run(main())
