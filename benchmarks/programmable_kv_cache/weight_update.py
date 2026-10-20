# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Change real Qwen weights through CUDA IPC while old KV remains in the Store."""

import argparse
import asyncio
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from torch.multiprocessing.reductions import reduce_tensor

from benchmarks.programmable_kv_cache.rollout import RolloutClient
from benchmarks.programmable_kv_cache.run_rollout_matrix import source_hashes
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferUpdateInfo


async def run(args):
    client = RolloutClient(args.base_url, "priority")
    report = {
        "passed": False,
        "requests": client.rows,
        "controls": client.controls,
        "scope": "real single-parameter IPC update and rollback; no optimizer step",
        "source_hashes": source_hashes(),
    }
    sync = HTTPVLLMWeightSyncClient(args.base_url)
    with gzip.open(args.trace, "rt") as file:
        prompt = json.load(file)["requests"][0]["prompt_ids"]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    name = next(
        n for n in index["weight_map"] if n.endswith("layers.0.input_layernorm.weight")
    )
    with safe_open(args.model / index["weight_map"][name], framework="pt") as file:
        original = file.get_tensor(name)
    changed = original + 0.125
    report["parameter"] = name
    report["original_sha256"] = hashlib.sha256(
        original.view(torch.uint8).numpy()
    ).hexdigest()
    report["changed_sha256"] = hashlib.sha256(
        changed.view(torch.uint8).numpy()
    ).hexdigest()

    async def update(tensor, version):
        for identifier in tuple(client.background):
            await client.drain(identifier)
        response = await client.http.post(
            "/pause", params={"mode": "wait", "clear_cache": False}
        )
        response.raise_for_status()
        await asyncio.to_thread(sync.start_weight_update)
        copies, handles = [], {}
        for device in range(torch.accelerator.device_count()):
            copy = tensor.to(device=f"cuda:{device}").contiguous()
            torch.accelerator.synchronize(device)
            copies.append(copy)
            handles[str(torch.cuda.get_device_properties(device).uuid)] = reduce_tensor(
                copy
            )[1]
        payload = IPCWeightTransferUpdateInfo(
            names=[name],
            dtype_names=[str(tensor.dtype).removeprefix("torch.")],
            shapes=[list(tensor.shape)],
            ipc_handles=[handles],
        )
        await asyncio.to_thread(sync.update_weights, asdict(payload))
        await asyncio.to_thread(sync.finish_weight_update, version)
        response = await client.http.post("/resume")
        response.raise_for_status()
        await client.initialize()
        return list(copies)

    try:
        await client.initialize()
        prompt = prompt[: client.boundary + 128]
        report["input_tokens"] = len(prompt)
        before_caps = client.capabilities
        before_version = (await client.http.get("/weight_info")).json()[
            "weight_version"
        ]
        report["before_weight_version"] = before_version
        before = await client.generate(
            "weight-before", prompt, {"temperature": 0, "max_tokens": 32}
        )
        logical = client.targets["weight-before"]
        resident = await client.call("residency", {"target": logical, "limit": 4096})
        target = {
            "kind": "hashes",
            "namespace": client.invocation["namespace"],
            "blocks": [
                {
                    k: row[k]
                    for k in ("block_hash", "group_id", "start_token", "end_token")
                }
                for row in resident["items"]
                if row["resident_copies"]
            ],
        }
        assert target["blocks"]
        stored = await client.action(
            "vllm.copy_to",
            {
                "target": target,
                "src": {"tier": "gpu"},
                "dst": {"tier": "mooncake"},
            },
        )
        assert stored["status"] == "completed", stored
        old_context = dict(client.invocation)
        await asyncio.to_thread(sync.init_weight_transfer_engine, {"packed": False})
        refs = await update(changed, "rollout-weights-changed")
        report["changed_capabilities"] = client.capabilities
        assert client.invocation["namespace"] != old_context["namespace"]
        stale = await client.http.post(
            "/v1/kv/control",
            json={
                "method": "associations",
                "invocation": old_context,
                "payload": {},
            },
        )
        assert stale.status_code == 409, stale.text
        report["stale_epoch"] = stale.json()
        incompatible = await client.http.post(
            "/v1/kv/control",
            json={
                "method": "residency",
                "invocation": client.invocation,
                "payload": {"target": target},
            },
        )
        assert incompatible.json()["error"]["code"] == "incompatible"
        relabeled = dict(target, namespace=client.invocation["namespace"])
        missing = await client.action(
            "vllm.copy_to",
            {
                "target": relabeled,
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu"},
            },
        )
        assert missing["status"] == "missing", missing
        changed_output = await client.generate(
            "weight-changed", prompt, {"temperature": 0, "max_tokens": 32}
        )
        assert changed_output["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert before["log_probs"] != changed_output["log_probs"]
        del refs
        refs = await update(original, before_version)
        assert client.invocation["namespace"] == before_caps["namespace"]
        recovered = await client.action(
            "vllm.copy_to",
            {
                "target": target,
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu"},
            },
        )
        assert recovered["status"] == "completed", recovered
        assert all(
            row["transfer_id"] is not None for row in recovered["target_outcomes"]
        )
        restored = await client.generate(
            "weight-restored", prompt, {"temperature": 0, "max_tokens": 32}
        )
        assert restored["token_ids"] == before["token_ids"]
        assert restored["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        report["passed"] = True
        report["old_store_objects_survived"] = True
        del refs
        print(
            json.dumps(
                {"passed": True, "parameter": name, "old_store_objects_survived": True}
            ),
            flush=True,
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        await client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.output, "wt") as file:
            json.dump(report, file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
