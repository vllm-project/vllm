# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``--dbo-scope``, which limits DBO to prefill or all batches."""

from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds

DECODE_THRESHOLD = 32
PREFILL_THRESHOLD = 512


def _config(dbo_scope: str, use_ubatching: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        use_ubatching=use_ubatching,
        num_ubatches=2,
        dbo_scope=dbo_scope,
        dbo_decode_token_threshold=DECODE_THRESHOLD,
        dbo_prefill_token_threshold=PREFILL_THRESHOLD,
    )


@pytest.mark.parametrize("dbo_scope", ["all", "prefill"])
def test_prefill_batches_follow_threshold(dbo_scope: str):
    config = _config(dbo_scope)
    assert check_ubatch_thresholds(config, PREFILL_THRESHOLD, uniform_decode=False)
    assert not check_ubatch_thresholds(
        config, PREFILL_THRESHOLD - 1, uniform_decode=False
    )


def test_scope_all_microbatches_decode():
    config = _config("all")
    assert check_ubatch_thresholds(config, DECODE_THRESHOLD, uniform_decode=True)
    assert not check_ubatch_thresholds(
        config, DECODE_THRESHOLD - 1, uniform_decode=True
    )


@pytest.mark.parametrize("num_tokens", [DECODE_THRESHOLD, 8192, 1 << 20])
def test_scope_prefill_never_microbatches_decode(num_tokens: int):
    config = _config("prefill")
    assert not check_ubatch_thresholds(config, num_tokens, uniform_decode=True)


@pytest.mark.parametrize("dbo_scope", ["all", "prefill"])
@pytest.mark.parametrize("uniform_decode", [True, False])
def test_no_microbatching_without_dbo(dbo_scope: str, uniform_decode: bool):
    config = _config(dbo_scope, use_ubatching=False)
    assert not check_ubatch_thresholds(config, 1 << 20, uniform_decode=uniform_decode)


def test_parallel_config_default_scope_is_all():
    assert ParallelConfig().dbo_scope == "all"


@pytest.mark.parametrize(
    ("cli_args", "expected"),
    [
        ([], "all"),
        (["--dbo-scope", "prefill"], "prefill"),
        (["--dbo-scope", "all"], "all"),
    ],
)
def test_cli_dbo_scope(cli_args: list[str], expected: str):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--enable-dbo", *cli_args])
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.enable_dbo
    assert engine_args.dbo_scope == expected


def test_cli_rejects_unknown_dbo_scope():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--dbo-scope", "decode"])


@pytest.mark.parametrize(
    ("dbo_scope", "expect_twin"), [("all", True), ("prefill", False)]
)
def test_ubatched_decode_graph_candidates(dbo_scope: str, expect_twin: bool):
    """Prefill scope captures no microbatched decode CUDA graphs."""
    manager = SimpleNamespace(
        ubatch_runner=object(),
        vllm_config=SimpleNamespace(parallel_config=_config(dbo_scope)),
    )
    # 64 decode requests with 3 speculative tokens each.
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=256, num_reqs=64
    )
    twin = CudaGraphManager._maybe_ubatch_twin(manager, desc)
    assert (twin is not None) == expect_twin
