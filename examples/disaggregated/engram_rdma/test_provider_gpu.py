# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small GPU test of the vLLM adapter's row layout and scratch ordering."""

import os
import sys
from types import ModuleType, SimpleNamespace

import torch

os.environ.pop("ENGRAM_SHARED_PREFETCH", None)


class FakeRuntime:
    def __init__(self):
        self.prefetch_enabled = False
        self.storage_layout = "row264"
        self.used = []

    def fetch(self, ids, layer, lo, hi):
        # Each row has a distinct exact FP8 value and one distinct power-of-two
        # scale, so head ordering, invalid ids and layer identity are observable.
        col = torch.arange(256, device=ids.device)
        block = torch.arange(8, device=ids.device)
        weights = ((ids.abs()[:, None] % 7) + layer + col % 4).to(torch.float8_e4m3fn)
        scales = (127 + (ids.abs()[:, None] + block) % 3).to(torch.uint8)
        return ids, weights, scales

    def record_consumed(self, layer):
        self.used.append(layer)

    def close(self):
        pass


mock = ModuleType("shared_model_adapter")
mock.ReaderRuntime = FakeRuntime
sys.modules[mock.__name__] = mock

import vllm_rdma_provider  # noqa: E402

vllm_rdma_provider.verify_checkpoint_samples = lambda: None
VllmRemoteReader = vllm_rdma_provider.VllmRemoteReader


def main():
    table = SimpleNamespace(
        dim=256,
        block_size=32,
        head_start=1,
        part_n_hash_cols=2,
        rdma_layer_id=1,
        vocab_start_idx=10,
        vocab_end_idx=20,
        num_embeddings=64,
    )
    ids = torch.tensor([[3, 10], [8, 19], [7, 20]], dtype=torch.int32, device="cuda")
    # The second local head is padded; output is a noncontiguous layer view.
    selected = torch.tensor([[10, -1], [19, -1], [20, -1]], device="cuda")
    valid = (selected >= 10) & (selected < 20)
    col = torch.arange(256, device="cuda")
    block = torch.arange(8, device="cuda")
    raw = ((selected.abs()[..., None] % 7) + 1 + col % 4).to(torch.float8_e4m3fn)
    scales = (127 + (selected.abs()[..., None] + block) % 3).to(torch.uint8)
    reader = VllmRemoteReader()
    for dtype, width in ((torch.bfloat16, 256), (torch.uint8, 264)):
        backing = torch.full((3, 2, 2, width), 37, dtype=dtype, device="cuda")
        out = backing[:, 0]
        reader.lookup(table, ids, out)
        if dtype == torch.bfloat16:
            expected = torch.where(
                valid[..., None],
                raw.float() * (2.0 ** (scales[..., col // 32].int() - 127)),
                0,
            )
            torch.testing.assert_close(out, expected.to(dtype))
        else:
            expected_values = torch.where(valid[..., None], raw.view(torch.uint8), 0)
            expected_scales = torch.where(valid[..., None], scales, 0)
            torch.testing.assert_close(out[..., :256], expected_values)
            torch.testing.assert_close(out[..., 256:], expected_scales)
        assert (backing[:, 1] == 37).all().item()
    torch.cuda.synchronize()
    assert reader.runtime.used == [1, 1]

    class FailingClose:
        def close(self):
            raise RuntimeError("retirement failed")

    retained = SimpleNamespace(runtime=FailingClose())
    vllm_rdma_provider._reader = retained
    try:
        vllm_rdma_provider.close_reader()
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed retirement was ignored")
    assert vllm_rdma_provider._reader is retained
    vllm_rdma_provider._reader = reader
    vllm_rdma_provider.close_reader()
    assert vllm_rdma_provider._reader is None
    print("PASS: BF16/packed rows, invalid IDs, TP-head padding, strided output")


if __name__ == "__main__":
    main()
