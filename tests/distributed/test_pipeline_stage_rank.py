# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ParallelConfig.pipeline_stage_rank`: one pipeline stage per instance."""

import contextlib
import os
import tempfile
from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.executor import Executor


class StagePPGroup(GroupCoordinator):
    """Reports the stage of this instance; the stages are not torch ranks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm.config import get_current_vllm_config

        parallel_config = get_current_vllm_config().parallel_config
        self.rank_in_group = parallel_config.pipeline_stage_rank
        self.world_size = parallel_config.pipeline_parallel_size


def _parallel_config(**kwargs) -> ParallelConfig:
    kwargs.setdefault("distributed_executor_backend", f"{__name__}.PPExecutor")
    return ParallelConfig(
        pipeline_parallel_size=4,
        pp_group_cls=f"{__name__}.StagePPGroup",
        **kwargs,
    )


def test_world_size_excludes_pipeline_stages():
    # Without pipeline_stage_rank the world would be 4 ranks (checked against
    # the visible GPUs); with it, only this stage's ranks.
    assert _parallel_config(pipeline_stage_rank=2).world_size == 1


def test_pipeline_stage_rank_validation():
    with pytest.raises(ValueError, match="pipeline_stage_rank"):
        _parallel_config(pipeline_stage_rank=4)
    with pytest.raises(ValueError, match="requires pp_group_cls"):
        ParallelConfig(pipeline_parallel_size=2, pipeline_stage_rank=1)


@pytest.mark.parametrize("backend", [None, "mp", "ray", "uni", "external_launcher"])
def test_pipeline_stage_rank_rejects_builtin_executors(backend):
    # The built-in executors expect all pipeline stages in this instance
    # (e.g. MultiprocExecutor asserts world_size == TP * PP * PCP).
    with pytest.raises(ValueError, match="requires a custom executor"):
        _parallel_config(
            pipeline_stage_rank=1,
            tensor_parallel_size=1,
            distributed_executor_backend=backend,
        )


def test_layer_range_follows_stage_rank():
    from vllm.config.model import ModelConfig

    model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: 32)
    for rank, expected in [(0, (0, 8)), (2, (16, 24)), (3, (24, 32))]:
        parallel_config = _parallel_config(pipeline_stage_rank=rank)
        assert (
            ModelConfig.get_layers_start_end_indices(model_config, parallel_config)
            == expected
        )


def test_cli_accepts_pipeline_stage_rank():
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        ["--pipeline-stage-rank", "1", "--pp-group-cls", "pkg.Group"]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.pipeline_stage_rank == 1
    assert engine_args.pp_group_cls == "pkg.Group"


def test_one_stage_per_process():
    """A single process runs stage 2 of 4: torch world of 1, PP group of 4."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    config = VllmConfig(parallel_config=_parallel_config(pipeline_stage_rank=2))
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=config.parallel_config.world_size,
                rank=0,
                distributed_init_method=f"file://{path}",
                local_rank=0,
                backend="gloo",
            )
            try:
                ensure_model_parallel_initialized(1, 4)
                pp_group = get_pp_group()
                assert type(pp_group) is StagePPGroup
                assert (pp_group.rank_in_group, pp_group.world_size) == (2, 4)
                assert get_tp_group().world_size == 1
            finally:
                cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def test_executor_qualname_declares_pp_support():
    engine_args = EngineArgs(
        pipeline_parallel_size=2,
        distributed_executor_backend=f"{__name__}.PPExecutor",
    )
    engine_args._check_feature_supported()


class PPExecutor(Executor):
    """Stands in for a plugin executor that drives the other stages."""

    supports_pp = True
