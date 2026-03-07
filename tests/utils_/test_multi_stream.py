# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.multi_stream import maybe_execute_in_parallel


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for multi-stream tests"
)
class TestMaybeExecuteInParallel:
    def setup_method(self):
        self.a = torch.randn(128, device="cuda")
        self.b = torch.randn(128, device="cuda")

    def test_sequential_mode(self):
        """Both functions run sequentially when aux_stream is None."""
        event0 = torch.cuda.Event()
        event1 = torch.cuda.Event()

        result0, result1 = maybe_execute_in_parallel(
            fn0=lambda: self.a + 1,
            fn1=lambda: self.b * 2,
            event0=event0,
            event1=event1,
            aux_stream=None,
        )

        torch.testing.assert_close(result0, self.a + 1)
        torch.testing.assert_close(result1, self.b * 2)

    def test_parallel_mode(self):
        """Both functions run on separate streams when aux_stream is given."""
        event0 = torch.cuda.Event()
        event1 = torch.cuda.Event()
        aux_stream = torch.cuda.Stream()

        result0, result1 = maybe_execute_in_parallel(
            fn0=lambda: self.a + 1,
            fn1=lambda: self.b * 2,
            event0=event0,
            event1=event1,
            aux_stream=aux_stream,
        )

        torch.testing.assert_close(result0, self.a + 1)
        torch.testing.assert_close(result1, self.b * 2)

    def test_parallel_matches_sequential(self):
        """Parallel and sequential modes produce identical results."""
        event0 = torch.cuda.Event()
        event1 = torch.cuda.Event()

        seq_r0, seq_r1 = maybe_execute_in_parallel(
            fn0=lambda: self.a + 1,
            fn1=lambda: self.b * 2,
            event0=event0,
            event1=event1,
            aux_stream=None,
        )

        par_r0, par_r1 = maybe_execute_in_parallel(
            fn0=lambda: self.a + 1,
            fn1=lambda: self.b * 2,
            event0=event0,
            event1=event1,
            aux_stream=torch.cuda.Stream(),
        )

        torch.testing.assert_close(seq_r0, par_r0)
        torch.testing.assert_close(seq_r1, par_r1)
