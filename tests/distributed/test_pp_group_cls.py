# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ParallelConfig.pp_group_cls` replaces the class of the PP group only."""

import contextlib
import os
import tempfile

import pytest

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    cleanup_dist_env_and_memory,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)


class RecordingPPGroup(GroupCoordinator):
    sent: list[dict] = []

    def send_tensor_dict(self, tensor_dict, dst=None, *args, **kwargs):
        self.sent.append(tensor_dict)
        return None


@contextlib.contextmanager
def _dist_env(pp_group_cls: str):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    config = VllmConfig(parallel_config=ParallelConfig(pp_group_cls=pp_group_cls))
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{path}",
                local_rank=0,
                backend="gloo",
            )
            try:
                initialize_model_parallel(1, 1)
                yield
            finally:
                cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def test_pp_group_cls_is_used_for_pp_group_only():
    with _dist_env(f"{__name__}.RecordingPPGroup"):
        pp_group = get_pp_group()
        assert type(pp_group) is RecordingPPGroup
        assert type(get_tp_group()) is GroupCoordinator
        assert pp_group.unique_name.startswith("pp")

        pp_group.send_tensor_dict({"x": 1})
        assert RecordingPPGroup.sent == [{"x": 1}]


def test_pp_group_cls_must_be_group_coordinator():
    with (
        pytest.raises(TypeError, match="GroupCoordinator subclass"),
        _dist_env("collections.OrderedDict"),
    ):
        pass
