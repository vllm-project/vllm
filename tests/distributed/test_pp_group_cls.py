# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ParallelConfig.pp_group_cls` replaces the class of the PP group only."""

import contextlib
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


def test_sibling_broadcast_default_uses_torch_distributed():
    with _dist_env(""):
        pp_group = get_pp_group()
        sibling = pp_group.make_sibling_device_group(group_desc="test")
        tensor = torch.arange(4)
        pp_group.sibling_broadcast(tensor, src=pp_group.last_rank, group=sibling)
        assert tensor.tolist() == [0, 1, 2, 3]


def test_pp_handler_broadcasts_through_pp_group(monkeypatch):
    """The V2 runner's sampled-token broadcast goes through the PP group, so
    a `pp_group_cls` can carry it over its own transport."""
    from vllm.v1.worker.gpu import pp_utils

    calls = []

    class FakePPGroup:
        is_last_rank = True
        last_rank = 1
        world_size = 2

        def make_sibling_device_group(self, group_desc=None):
            return "sibling"

        def sibling_broadcast(self, tensor, src, group):
            calls.append((tuple(tensor.shape), src, group))

    monkeypatch.setattr(pp_utils, "get_pp_group", FakePPGroup)
    monkeypatch.setattr(pp_utils.torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(pp_utils.torch.cuda, "Stream", lambda device: None)
    monkeypatch.setattr(
        pp_utils.torch.cuda, "stream", lambda s: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    handler = pp_utils.PPHandler(max_num_reqs=4, num_speculative_steps=0, device="cpu")
    handler.main_stream = handler.broadcast_stream = _NullStream()

    batch = SimpleNamespace(
        num_computed_tokens_np=np.array([4, 4]),
        prefill_len_np=np.array([4, 4]),
        num_scheduled_tokens=np.array([1, 1]),
        max_seq_len_np=None,
        is_prefilling_np=np.array([False, False]),
        num_draft_tokens=0,
    )
    sampled = torch.zeros(2, 1, dtype=torch.int64)
    handler.broadcast(
        sampled,
        torch.ones(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        batch,
    )
    assert calls == [((2, 1), 1, "sibling"), ((2, 2), 1, "sibling")]


class _NullStream:
    def wait_stream(self, stream):
        pass

    def synchronize(self):
        # PPHandler.broadcast() synchronizes the main stream on XPU.
        pass
