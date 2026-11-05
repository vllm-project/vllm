# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check split-page gathers, padding, invalid sources, and compact miss counts."""

import argparse
import json
from pathlib import Path

import torch


def check(ops, width, value_width, block=64):
    page = block * width
    stride = ((page + 575) // 576) * 576
    records = (torch.arange(3 * page).view(3 * block, width) % 251).to(torch.uint8)
    host = torch.empty((3, block, width), dtype=torch.uint8, pin_memory=True)
    raw = torch.full((5 * stride + 64,), 253, dtype=torch.uint8, device="cuda")
    hot = raw.as_strided((5, block, width), (stride, width, 1), storage_offset=32)

    def planes(storage, pages, page_stride, offset):
        if not value_width:
            return [
                storage.as_strided(
                    (pages, block, width),
                    (page_stride, width, 1),
                    storage_offset=offset,
                )
            ]
        return [
            storage.as_strided(
                (pages, block, value_width),
                (page_stride, value_width, 1),
                storage_offset=offset,
            ),
            storage.as_strided(
                (pages, block, width - value_width),
                (page_stride, width - value_width, 1),
                storage_offset=offset + block * value_width,
            ),
        ]

    begin = 0
    for view in planes(host, 3, page, 0):
        count = view.shape[-1]
        view.copy_(records[:, begin : begin + count].reshape_as(view))
        begin += count

    def tensor(values):
        return torch.tensor([values], dtype=torch.int32, device="cuda")

    sources = [0, block - 1, block, 2 * block - 1, 2 * block, 3 * block - 1, 3 * block]
    destinations = [
        0,
        block - 1,
        block,
        2 * block,
        3 * block - 1,
        4 * block - 1,
        5 * block - 1,
    ]
    ops.hisparse_gather_plan(
        host.view(-1, width),
        hot,
        tensor(sources),
        tensor(destinations),
        torch.ones((1, len(sources)), dtype=torch.int32, device="cuda"),
        None,
        None,
        0,
        value_width,
    )
    expected = torch.full((5 * block, width), 253, dtype=torch.uint8)
    for src, dst in zip(sources, destinations):
        expected[dst] = records[src] if src < len(records) else 0

    def verify():
        torch.accelerator.synchronize()
        copied = raw.cpu()
        actual = torch.cat(
            [p.reshape(-1, p.shape[-1]) for p in planes(copied, 5, stride, 32)], dim=1
        )
        assert torch.equal(actual, expected)
        assert torch.all(copied[:32] == 253)
        assert torch.all(copied[-32:] == 253)
        for i in range(5):
            assert torch.all(
                copied[32 + i * stride + page : 32 + (i + 1) * stride] == 253
            )

    verify()
    # The padded fourth entry must be ignored by the compact count.
    ops.hisparse_gather_compact(
        host.view(-1, width),
        hot,
        tensor([3 * block - 1, 0, 3 * block, 17]),
        tensor([1, block - 2, block + 1, 17]),
        tensor([3]).flatten(),
        value_width,
    )
    expected[1], expected[block - 2], expected[block + 1] = (
        records[3 * block - 1],
        records[0],
        0,
    )
    verify()
    return dict(
        row_bytes=width,
        value_bytes=value_width,
        block_size=block,
        rows_checked=5 * block,
        guards_exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    results = [
        check(torch.ops._C_dsv41_hisparse, 656, 0),
        check(torch.ops._C_dsv41_hisparse, 584, 576),
        check(torch.ops._C_dsv41_hisparse, 584, 576, 32),
    ]
    result = dict(passed=True, checks=results, gpu=torch.cuda.get_device_name())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
