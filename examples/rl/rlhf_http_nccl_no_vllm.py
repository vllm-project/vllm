# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL weight sync over HTTP from a trainer that never imports vLLM.

The trainer uses only `torch`, `requests`, `safetensors`, `pybase64` and the
stdlib, and speaks the wire protocol in
`docs/training/weight_transfer/nccl.md` ("Trainer Without vLLM over HTTP"):
`nccl_unique_id_b64` rendezvous, a one-element warm-up `all_reduce`, then
`/start_weight_update` -> `/update_weights` (concurrent with the rank-0
broadcast) -> `/finish_weight_update`.

The NCCL communicator is a plain `torch.distributed.ProcessGroupNCCL` over a
private `TCPStore` that only the trainer joins. Its `ncclUniqueId` is read back
from store key "0", which is a torch implementation detail.

Start a server on GPUs 0-1 (the weights it starts with are discarded):

    VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-0.6B \
        --tensor-parallel-size 2 --load-format dummy --enforce-eager \
        --weight-transfer-config '{"backend": "nccl"}'

Then run the trainer on GPU 2 (`--packed` for packed broadcasting):

    CUDA_VISIBLE_DEVICES=2 python examples/rl/rlhf_http_nccl_no_vllm.py \
        --checkpoint /path/to/Qwen3-0.6B
"""

import sys


class _BlockVLLM:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "vllm":
            raise ImportError(f"this trainer must not import {name!r}")
        return None


sys.meta_path.insert(0, _BlockVLLM())

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import threading  # noqa: E402

import pybase64 as base64  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--checkpoint", required=True, help="dir with *.safetensors")
parser.add_argument("--packed", action="store_true")
parser.add_argument("--packed-buffer-size-bytes", type=int, default=64 << 20)
parser.add_argument("--store-port", type=int, default=29612)
args = parser.parse_args()


def post(path: str, body: dict | None = None) -> dict:
    r = requests.post(args.url + path, json=body or {}, timeout=600)
    r.raise_for_status()
    return r.json()


class PostInBackground(threading.Thread):
    """An HTTP call that only returns once this process joins the collective."""

    def __init__(self, path: str, body: dict) -> None:
        super().__init__()
        self.path, self.body = path, body
        self.error: BaseException | None = None
        self.start()

    def run(self) -> None:
        try:
            post(self.path, self.body)
        except BaseException as e:
            self.error = e

    def result(self) -> None:
        self.join()
        if self.error is not None:
            raise self.error


def generate(prompts: list[str]) -> list[str]:
    model = requests.get(args.url + "/v1/models", timeout=60).json()["data"][0]["id"]
    body = {"model": model, "prompt": prompts, "max_tokens": 16, "temperature": 0}
    choices = post("/v1/completions", body)["choices"]
    return [c["text"] for c in sorted(choices, key=lambda c: c["index"])]


def main() -> None:
    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device)
    weights: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(f"{args.checkpoint}/*.safetensors")):
        weights.update(load_file(path, device="cuda:0"))
    names = sorted(weights)

    prompts = ["The capital of France is", "def fibonacci(n):"]
    print("before:", generate(prompts))

    # Trainer is NCCL rank 0; vLLM worker i is rank 1 + i.
    world_size = 1 + requests.get(args.url + "/get_world_size").json()["world_size"]
    store = dist.TCPStore(
        "127.0.0.1", args.store_port, is_master=True, wait_for_workers=False
    )
    pg = dist.ProcessGroupNCCL(store, 0, world_size, dist.ProcessGroupNCCL.Options())
    connect = threading.Thread(target=pg.eager_connect_single_device, args=(device,))
    connect.start()  # publishes the unique id, then waits in ncclCommInitRank
    reader = dist.TCPStore("127.0.0.1", args.store_port, is_master=False)
    reader.wait(["0"], datetime.timedelta(seconds=60))
    unique_id = reader.get("0")
    assert len(unique_id) == 128

    init = PostInBackground(
        "/init_weight_transfer_engine",
        {
            "init_info": {
                "nccl_unique_id_b64": base64.b64encode(unique_id).decode(),
                "rank_offset": 1,
                "world_size": world_size,
                "packed": args.packed,
                "packed_buffer_size_bytes": args.packed_buffer_size_bytes,
            }
        },
    )
    connect.join()
    pg.allreduce([torch.zeros(1, device=device)]).wait()  # vLLM's warm-up
    init.result()

    def broadcast(t: torch.Tensor) -> None:
        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        pg.broadcast([t], opts).wait()

    post("/start_weight_update")
    update_info = {
        "names": names,
        "dtype_names": [str(weights[n].dtype).removeprefix("torch.") for n in names],
        "shapes": [list(weights[n].shape) for n in names],
    }
    update = PostInBackground("/update_weights", {"update_info": update_info})
    if not args.packed:
        for n in names:
            broadcast(weights[n].contiguous())
    else:
        # Same chunking as the worker: a chunk closes right after the tensor
        # that pushes it past the buffer size.
        chunk: list[torch.Tensor] = []
        size = 0
        for n in names:
            chunk.append(weights[n].contiguous().view(-1).view(torch.uint8))
            size += chunk[-1].numel()
            if size > args.packed_buffer_size_bytes:
                broadcast(torch.cat(chunk))
                chunk, size = [], 0
        if chunk:
            broadcast(torch.cat(chunk))
    torch.accelerator.synchronize()
    update.result()
    post("/finish_weight_update", {"weight_version": None})

    print("after: ", generate(prompts))
    post("/shutdown_weight_transfer_engine")
    pg.shutdown()

    assert not [m for m in sys.modules if m.split(".")[0] == "vllm"]


if __name__ == "__main__":
    main()
