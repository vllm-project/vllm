# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import unittest
from unittest.mock import patch

import torch

from vllm.model_executor.kernels.attention.dsa.candidate_gather import (
    candidate_gather_decode_available,
    candidate_gather_decode_profitable,
    candidate_gather_score_decode,
    candidate_gather_workspace_specs,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.v1.worker import workspace


class CandidateGatherForceTest(unittest.TestCase):
    def test_force_bypasses_workload_gates_not_enable_switch(self):
        with patch.dict(
            os.environ,
            {
                "VLLM_DSA_CANDIDATE_GATHER_FORCE": "0",
                "VLLM_DSA_CANDIDATE_GATHER_DECODE": "0",
                "VLLM_DSA_CANDIDATE_GATHER_MIN_ROWS": "16",
                "VLLM_DSA_CANDIDATE_GATHER_MIN_WORK_RATIO": "1500",
            },
        ):
            self.assertFalse(candidate_gather_decode_profitable(6, 6, 2048, 8))
            os.environ["VLLM_DSA_CANDIDATE_GATHER_FORCE"] = "1"
            self.assertTrue(candidate_gather_decode_profitable(6, 6, 2048, 8))
            self.assertFalse(
                candidate_gather_decode_available(False, 1, False, 8, 2048, 512, "auto")
            )


class CandidateGatherGraphTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if not current_platform.is_device_capability_family(90) or not has_deep_gemm():
            self.skipTest("SM90 and DeepGEMM required")

    def test_persistent_topk_capture_bound(self):
        from vllm.model_executor.layers.indexer_topk import get_indexer_topk

        with patch.object(workspace, "_manager", None):
            device = torch.device("cuda", torch.cuda.current_device())
            workspace.init_workspace_manager(device)
            rows, width, topk = 6, 16384, 512
            logits = torch.arange(width, dtype=torch.float32, device=device)
            logits = logits.repeat(rows, 1)
            lens = torch.full((rows, 1), 6, dtype=torch.int32, device=device)
            out = torch.empty(rows, topk, dtype=torch.int32, device=device)
            topk_op = get_indexer_topk("persistent")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    topk_op(logits, lens, 1, out, topk, width)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            workspace.lock_workspace()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                topk_op(logits, lens, 1, out, topk, 6)
            lengths = [16000, 0, 6, 512, 513, 8192]
            lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device)[:, None])
            graph.replay()
            expected = torch.full_like(out, -1)
            for row, length in enumerate(lengths):
                count = min(length, topk)
                expected[row, :count] = torch.arange(
                    length - count, length, device=device
                )
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out.sort(1).values, expected.sort(1).values))
            del graph
            workspace.reset_workspace_manager()

    def test_short_capture_long_replay(self):
        with (
            patch.dict(
                os.environ,
                {
                    "VLLM_DSA_CANDIDATE_GATHER_DECODE": "1",
                    "VLLM_DSA_CANDIDATE_GATHER_FORCE": "1",
                    "VLLM_DSA_CANDIDATE_GATHER_MIN_ROWS": "16",
                    "VLLM_DSA_CANDIDATE_GATHER_MIN_WORK_RATIO": "1500",
                },
            ),
            patch.object(workspace, "_manager", None),
        ):
            for rows, backend in ((6, "persistent"), (54, "auto")):
                with self.subTest(rows=rows, backend=backend):
                    self._check_replay(rows, backend)

    def _check_replay(self, rows, backend):
        torch.manual_seed(12)
        device = torch.device("cuda", torch.cuda.current_device())
        workspace.init_workspace_manager(device)
        block_size, head_dim, num_candidates, cbk, topk = 64, 128, 2048, 8, 512
        context = 32768
        num_blocks = context // block_size
        page_bytes = block_size * (head_dim + 4)
        backing = torch.zeros(
            (num_blocks, page_bytes + 64), dtype=torch.uint8, device=device
        )
        fp8 = current_platform.fp8_dtype()
        values = torch.randn(num_blocks, block_size * head_dim, device=device).to(fp8)
        scales = torch.ones(num_blocks, block_size, device=device)
        backing[:, : block_size * head_dim] = values.view(torch.uint8)
        backing[:, block_size * head_dim : page_bytes] = scales.view(torch.uint8)
        cache = backing[:, :page_bytes].view(num_blocks, block_size, head_dim + 4)
        self.assertFalse(cache.is_contiguous())
        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
        candidates = torch.full(
            (1, num_candidates), -1, device=device, dtype=torch.int32
        )
        seq_lens = torch.full((rows, 1), 6, device=device, dtype=torch.int32)
        q = torch.randn(rows, 32, head_dim, device=device).to(fp8)
        weights = torch.rand(rows, 32, device=device)
        out = torch.empty(rows, topk, device=device, dtype=torch.int32)
        workspace.current_workspace_manager().get_simultaneous(
            *candidate_gather_workspace_specs(rows, num_candidates, cbk, head_dim, topk)
        )
        # Pass expanded row views with stable storage, not new allocations.
        bt_rows = block_table.expand(rows, -1)
        cand_rows = candidates.expand(rows, -1)

        def run(host_context):
            if not candidate_gather_decode_available(
                False, 1, False, cbk, num_candidates, topk, backend
            ) or not candidate_gather_decode_profitable(
                host_context, rows, num_candidates, cbk
            ):
                raise AssertionError("force must select gather even for dummy context")
            candidate_gather_score_decode(
                q,
                cache,
                weights,
                seq_lens,
                bt_rows,
                cand_rows,
                cbk,
                1,
                topk,
                backend,
                out,
                head_dim,
                host_context,
                fp8,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            seq_lens.fill_(context)
            candidates.copy_(torch.arange(num_candidates, device=device)[None])
            run(context)
            seq_lens.fill_(6)
            candidates.fill_(-1)
            candidates[:, 0] = 0
            for _ in range(3):
                run(6)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        workspace.lock_workspace()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(6)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal((out >= 0).sum(1), torch.full_like(seq_lens[:, 0], 6))
        )

        for length in (context, 0, context - 3):
            seq_lens.fill_(length)
            candidates.copy_(
                torch.randperm(context // cbk, device=device)[:num_candidates][None]
            )
            block_table.copy_(torch.randperm(num_blocks, device=device)[None])
            if length:
                seq_lens[0] = 0
                seq_lens[1] = 6
            else:
                candidates.fill_(-1)
            graph.replay()
            actual = out.clone()
            run(length)
            expected = out.clone()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual.sort(1).values, expected.sort(1).values))
            self.assertTrue(bool((actual[0] == -1).all()))
            if length:
                self.assertTrue(bool(((actual[2:] >= 0).sum(1) == topk).all()))
                self.assertGreater(int(actual[2:].max()), 6)
            else:
                self.assertTrue(bool((actual == -1).all()))
        del graph
        torch.cuda.synchronize()
        workspace.reset_workspace_manager()


if __name__ == "__main__":
    unittest.main()
