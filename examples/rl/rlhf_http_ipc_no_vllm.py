# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA IPC weight sync over HTTP from a trainer that never imports vLLM.

The trainer uses only `torch`, `requests`, `safetensors`, `pybase64` and the
stdlib, and speaks the wire protocol in
`docs/training/weight_transfer/ipc.md` ("Trainer Without vLLM over HTTP"):
it exports its tensors with `torch.multiprocessing.reductions.reduce_tensor`
and sends the handles, pickled and base64-encoded, in `ipc_handles_pickled`.

IPC requires colocation: this process holds a full copy of the weights on every
GPU the server uses, and keys each handle by that GPU's UUID. A multi-process
trainer would instead gather the per-rank `{uuid: args}` dicts to one rank.

Start a server; IPC handles are pickled, so it needs
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` (the weights it starts with are discarded):

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
        --load-format dummy --enforce-eager --gpu-memory-utilization 0.4 \
        --weight-transfer-config '{"backend": "ipc"}'

Then run the trainer on the same GPU(s) (`--packed` for chunked transfer):

    CUDA_VISIBLE_DEVICES=0 python examples/rl/rlhf_http_ipc_no_vllm.py \
        --checkpoint /path/to/Qwen3-0.6B

For a TP=2 server on GPUs 0-1, use `CUDA_VISIBLE_DEVICES=0,1` on both sides and
pass `--num-gpus 2` to the trainer.
"""

import sys


class _BlockVLLM:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "vllm":
            raise ImportError(f"this trainer must not import {name!r}")
        return None


sys.meta_path.insert(0, _BlockVLLM())

import argparse  # noqa: E402
import glob  # noqa: E402
import pickle  # noqa: E402

import pybase64 as base64  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch.multiprocessing.reductions import reduce_tensor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--checkpoint", required=True, help="dir with *.safetensors")
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--packed", action="store_true")
parser.add_argument("--packed-buffer-size-bytes", type=int, default=512 << 20)
args = parser.parse_args()


def post(path: str, body: dict | None = None) -> dict:
    r = requests.post(args.url + path, json=body or {}, timeout=600)
    r.raise_for_status()
    return r.json()


def generate(prompts: list[str]) -> list[str]:
    model = requests.get(args.url + "/v1/models", timeout=60).json()["data"][0]["id"]
    body = {"model": model, "prompt": prompts, "max_tokens": 16, "temperature": 0}
    choices = post("/v1/completions", body)["choices"]
    return [c["text"] for c in sorted(choices, key=lambda c: c["index"])]


# Weights are {gpu_uuid: {name: tensor}}: one full copy per colocated GPU. The
# UUID is str(torch.cuda.get_device_properties(i).uuid), with no "GPU-" prefix.
Weights = dict[str, dict[str, torch.Tensor]]


def update_weights(
    weights: Weights,
    names: list[str],
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
    tensor_sizes: list[int] | None = None,
) -> None:
    w = next(iter(weights.values()))
    update_info = {
        "names": names,
        "dtype_names": [str(w[n].dtype).removeprefix("torch.") for n in names],
        "shapes": [list(w[n].shape) for n in names],
        "ipc_handles_pickled": base64.b64encode(pickle.dumps(ipc_handles)).decode(),
    }
    if tensor_sizes is not None:
        update_info["tensor_sizes"] = tensor_sizes
    post("/update_weights", {"update_info": update_info})


def send_unpacked(weights: Weights, names: list[str]) -> list[torch.Tensor]:
    """One /update_weights call; one {uuid: args} dict per parameter."""
    copies = {u: {n: w[n].contiguous() for n in names} for u, w in weights.items()}
    handles = [{u: reduce_tensor(copies[u][n])[1] for u in copies} for n in names]
    update_weights(weights, names, handles)
    return [t for c in copies.values() for t in c.values()]


def send_packed(weights: Weights, names: list[str], cap: int) -> list[torch.Tensor]:
    """One /update_weights call per chunk; one {uuid: args} dict per chunk."""
    buffers = {
        u: torch.empty(cap, dtype=torch.uint8, device=w[names[0]].device)
        for u, w in weights.items()
    }
    handle = {u: reduce_tensor(b)[1] for u, b in buffers.items()}  # export once
    chunk: list[str] = []
    sizes: list[int] = []

    def flush() -> None:
        for b in buffers.values():
            torch.accelerator.synchronize(b.device)  # copies must land before the read
        update_weights(weights, chunk, handle, sizes)
        chunk.clear()
        sizes.clear()

    for n in names:
        nbytes = next(iter(weights.values()))[n].nbytes
        assert nbytes <= cap, f"{n} ({nbytes} B) exceeds the buffer"
        if chunk and sum(sizes) + nbytes > cap:  # close *before* overflowing
            flush()
        offset = sum(sizes)
        for u, w in weights.items():
            flat = w[n].contiguous().view(-1).view(torch.uint8)
            buffers[u][offset : offset + nbytes].copy_(flat)
        chunk.append(n)
        sizes.append(nbytes)
    if chunk:
        flush()
    return list(buffers.values())


def main() -> None:
    weights: Weights = {}
    for i in range(args.num_gpus):
        uuid = str(torch.cuda.get_device_properties(i).uuid)
        weights[uuid] = {}
        for path in sorted(glob.glob(f"{args.checkpoint}/*.safetensors")):
            weights[uuid].update(load_file(path, device=f"cuda:{i}"))
    names = sorted(next(iter(weights.values())))

    prompts = ["The capital of France is", "def fibonacci(n):"]
    print("before:", generate(prompts))

    post("/init_weight_transfer_engine", {"init_info": {"packed": args.packed}})
    post("/start_weight_update")
    if args.packed:
        exported = send_packed(weights, names, args.packed_buffer_size_bytes)
    else:
        exported = send_unpacked(weights, names)
    post("/finish_weight_update", {"weight_version": None})
    # The worker reads the exported memory until finish; only then release it.
    del exported
    torch.cuda.ipc_collect()

    print("after: ", generate(prompts))
    assert not [m for m in sys.modules if m.split(".")[0] == "vllm"]


if __name__ == "__main__":
    main()
