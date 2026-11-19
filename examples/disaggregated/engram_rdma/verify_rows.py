# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a bundled reader's remote rows with the local checkpoint."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.distributed.engram_share.rdma.owner import inspect_checkpoint
from vllm.distributed.engram_share.rdma.provider import VllmRemoteReader
from vllm.distributed.engram_share.rdma.runtime import OwnerClient


def checkpoint_row(manifest, layer, row):
    parts = []
    for kind, width in (("weight", 256), ("scale", 8)):
        spec = next(
            item
            for item in manifest["specs"]
            if item["layer"] == layer and item["kind"] == kind
        )
        shard = manifest["shards"][spec["shard"]]
        offset = shard["data_start"] + spec["data_offsets"][0] + row * width
        with open(shard["path"], "rb", buffering=0) as file:
            data = os.pread(file.fileno(), width, offset)
        if len(data) != width:
            raise ValueError("short checkpoint row")
        parts.append(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    return torch.cat(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reader-port-base", type=int, required=True)
    args = parser.parse_args()
    os.environ.update(
        ENGRAM_SHARED_READY=str(args.ready),
        ENGRAM_SHARED_LIBRARY=str(args.library),
        ENGRAM_SHARED_MODEL=str(args.model),
        ENGRAM_SHARED_READER_PORT_BASE=str(args.reader_port_base),
        ENGRAM_SHARED_MAX_QUERIES="2048",
    )
    ready = json.loads(args.ready.read_text())
    client = OwnerClient(
        ready["host"], ready["port"], ready["checkpoint_id"], ready["generation"]
    )
    initial_leases = client.status()["leases"]
    manifest = inspect_checkpoint(args.model)
    reader = VllmRemoteReader(args.model)
    queued = []
    try:
        for layer, rows in zip((1, 14), manifest["rows"]):
            table = SimpleNamespace(
                dim=256,
                block_size=32,
                head_start=0,
                part_n_hash_cols=2,
                rdma_layer_id=layer,
                vocab_start_idx=0,
                vocab_end_idx=rows,
                num_embeddings=rows,
            )
            choices = [0, 1, rows // 2, rows - 1, -1, rows]
            expected_rows = [
                checkpoint_row(manifest, layer, row)
                if 0 <= row < rows
                else torch.zeros(264, dtype=torch.uint8)
                for row in choices
            ]
            for count in (6, 514):
                indices = [i % (6 if count == 6 else 4) for i in range(count)]
                ids = torch.tensor(
                    [choices[i] for i in indices], dtype=torch.int32, device="cuda"
                ).view(-1, 2)
                output = torch.empty(
                    (count // 2, 2, 264), dtype=torch.uint8, device="cuda"
                )
                reader.lookup(table, ids, output)
                expected = torch.stack([expected_rows[i] for i in indices]).view_as(
                    output.cpu()
                )
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                if count == 514:
                    assert int(reader.runtime.stats.cpu()[4]) == 514
        print("PASS: both layers, padding and 514-query boundary match checkpoint")

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for index in range(16):
            layer_index = index % 2
            layer = (1, 14)[layer_index]
            rows = manifest["rows"][layer_index]
            values = [index, index + 1, rows // 2, rows - 1, -1, rows, 2, 3]
            table = SimpleNamespace(
                dim=256,
                block_size=32,
                head_start=0,
                part_n_hash_cols=2,
                rdma_layer_id=layer,
                vocab_start_idx=0,
                vocab_end_idx=rows,
                num_embeddings=rows,
            )
            with torch.cuda.stream(streams[(index // 2) % 2]):
                ids = torch.tensor(values, dtype=torch.int32, device="cuda").view(4, 2)
                output = torch.empty((4, 2, 264), dtype=torch.uint8, device="cuda")
                reader.lookup(table, ids, output)
            queued.append((layer, rows, values, ids, output))
    finally:
        # Retire queued work before returning the lease; no preceding CUDA sync.
        reader.close()
    for layer, rows, values, _, output in queued:
        expected = torch.stack(
            [
                checkpoint_row(manifest, layer, row)
                if 0 <= row < rows
                else torch.zeros(264, dtype=torch.uint8)
                for row in values
            ]
        ).view_as(output.cpu())
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    try:
        reader.lookup(table, ids, output)
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed reader accepted a lookup")
    if client.status()["leases"] != initial_leases:
        raise RuntimeError("owner lease count did not return to its initial value")
    print("PASS: 16 alternating-stream reads retired before owner lease release")


if __name__ == "__main__":
    main()
