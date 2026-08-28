import os

import torch

from vllm.v1.simple_kv_offload.legomem_backend import LegoMemBackend


def main() -> None:
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    active = torch.arange(512, dtype=torch.int16).view(4, 128)
    expected = active[2].clone()
    backend = LegoMemBackend()
    backend.init(
        {"kv": active},
        "/home/ubuntu/legomem/lib/liblegomem_kv.so",
        "127.0.0.1",
        9999,
        rank,
        16,
        256 * 1024 * 1024,
        256 * 1024 * 1024 - 64,
    )
    events = []
    backend.launch_copy([2], [7], True, 1, events)
    active[2].zero_()
    backend.launch_copy([7], [2], False, 2, events)
    passed = torch.equal(active[2], expected)
    print(
        f"LEGOMEM_BACKEND_ROUNDTRIP={'PASS' if passed else 'FAIL'} "
        f"rank={rank} target={(rank + 1) % 16} "
        f"bytes_written={backend.bytes_written} bytes_read={backend.bytes_read}"
    )
    backend.shutdown()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
