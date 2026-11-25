# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
from contextlib import contextmanager, nullcontext

import pytest
import torch

from vllm.config import (
    DeviceConfig,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.kernel import IrOpPriorityConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.minicpm5_moe.xpu.specialized_decode.config import SHAPES, configure
from vllm.models.minicpm5_moe.xpu.specialized_decode.sampling import enabled
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minicpm5_moe import MiniCPM5MoEConfig


def _minicpm5_config(
    tmp_path, *, mono=False, options=None, tp=1, masked_token_ids=None
):
    model_path = tmp_path / "model"
    MiniCPM5MoEConfig(
        architectures=["MiniCPM5MoEForCausalLM"], **SHAPES
    ).save_pretrained(model_path)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(model_path),
            dtype="float16",
            max_model_len=2048,
            enforce_eager=True,
            quantization="fp8_per_tensor",
            skip_tokenizer_init=True,
            masked_token_ids=masked_token_ids,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=tp),
        kernel_config=KernelConfig(
            monokernel=mono,
            ir_op_priority=IrOpPriorityConfig(
                rms_norm=["vllm_c"], fused_add_rms_norm=["vllm_c"]
            ),
        ),
        additional_config=options or {},
    )


@pytest.mark.parametrize(
    ("tp", "requested", "masked_token_ids", "expected"),
    [
        (2, False, None, False),
        (2, True, None, True),
        (1, True, None, False),
        (2, True, [129], False),
    ],
)
def test_candidate_sampling_respects_shared_token_mask(
    tmp_path, tp, requested, masked_token_ids, expected
):
    config = _minicpm5_config(
        tmp_path,
        options={"minicpm5_candidate_sampling": requested},
        tp=tp,
        masked_token_ids=masked_token_ids,
    )
    assert enabled(config) is expected


@contextmanager
def _cpu_model_parallel(config, tmp_path):
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=(tmp_path / "rendezvous").as_uri(),
        )
        initialize_model_parallel(1, 1)
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_non_battlemage_rejects_mono_before_provider_import(tmp_path, monkeypatch):
    config = _minicpm5_config(tmp_path, mono=True)
    with _cpu_model_parallel(config, tmp_path):
        monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
        monkeypatch.setattr(torch.xpu, "current_device", lambda: 4)
        queried = []

        def is_bmg(index):
            queried.append(index)
            return False

        monkeypatch.setattr(torch.ops._xpu_C, "is_bmg", is_bmg, raising=False)
        import_module = builtins.__import__

        def reject_provider(name, *args, **kwargs):
            if name == "vllm_xpu_kernels.minicpm5_mono":
                raise AssertionError("Mono provider imported before XPU qualification")
            return import_module(name, *args, **kwargs)

        with monkeypatch.context() as refusing_import:
            refusing_import.setattr(builtins, "__import__", reject_provider)
            with pytest.raises(ValueError, match="requires a Battlemage"):
                configure(config)
            config.kernel_config.monokernel = False
            config.additional_config = {}
            assert configure(config) is None
        assert queried == [4]


@pytest.mark.parametrize("is_checkpoint_format", [False, True])
@pytest.mark.parametrize("mode", ["mono", "ordinary", "split"])
def test_mini_mono_rejects_reload_before_kv_loan(
    tmp_path, monkeypatch, mode, is_checkpoint_format
):
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.xpu_worker import XPUWorker

    options = {"minicpm5_dspark_split_kv": True} if mode == "split" else {}
    config = _minicpm5_config(tmp_path, mono=mode == "mono", options=options)
    config.device_config = DeviceConfig(device="xpu")
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    worker = XPUWorker(config, local_rank=0, rank=0, distributed_init_method="")
    model = torch.nn.Module()
    model.model = torch.nn.Linear(1, 1, bias=False)
    if mode == "mono":
        model.model.specialized_decode = object()
    previous = model.model.weight.detach().clone()
    events = []

    def checkpoint():
        events.append("read")
        yield "model.weight", previous + 1

    def loan(*args, **kwargs):
        events.append("loan")
        return nullcontext()

    class Storage:
        def get_model(self):
            return model

        def reload_weights(self, iterator, path, checkpoint_format):
            assert path == "checkpoint" and checkpoint_format == is_checkpoint_format
            with torch.no_grad():
                for name, value in iterator:
                    model.get_parameter(name).copy_(value)
            events.append("write")

    monkeypatch.setattr(gpu_worker, "kv_cache_lent_for_weight_update", loan)
    worker.model_runner = Storage()
    if mode == "mono":
        with pytest.raises(NotImplementedError, match="MiniCPM5 XPU mono"):
            worker.reload_weights(checkpoint(), "checkpoint", is_checkpoint_format)
        assert events == []
        torch.testing.assert_close(model.model.weight, previous, atol=0, rtol=0)
    else:
        worker.reload_weights(checkpoint(), "checkpoint", is_checkpoint_format)
        assert events == (["loan"] if is_checkpoint_format else []) + ["read", "write"]
        torch.testing.assert_close(model.model.weight, previous + 1, atol=0, rtol=0)
