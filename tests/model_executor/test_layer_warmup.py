# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import nn

from vllm.model_executor.warmup.layer_warmup import LayerWarmup, run_layer_warmups


class Warmed(nn.Module, LayerWarmup):
    calls: list[tuple[list, object]] = []

    @classmethod
    def warmup_jit_kernels(cls, layers, worker):
        cls.calls.append((layers, worker))


class WarmedChild(Warmed):
    pass


class NotWarmed(nn.Module, LayerWarmup):
    pass


def test_hook_runs_once_per_overriding_class_with_all_instances():
    a, b, c = Warmed(), WarmedChild(), Warmed()
    model = nn.Sequential(a, NotWarmed(), b, nn.Sequential(c))
    Warmed.calls.clear()

    run_layer_warmups(model, "warmup_jit_kernels", "worker")
    run_layer_warmups(model, "warmup_model_runs", "worker")

    assert Warmed.calls == [([a, b, c], "worker")]
