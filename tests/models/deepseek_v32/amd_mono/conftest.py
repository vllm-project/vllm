# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures of the GLM-5.2 MonoKernel (ROCm, mono/) CPU tests. MonoLive and
Glm5MonoDecode are built bare (``__new__``) around fake kernel ops (no GPU,
no FlyDSL)."""

import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import envs as E


@pytest.fixture
def no_device_sync(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def failstop_latch(monkeypatch):
    """No latched output-rank fail-stop, and a fail-stop does not arm the exit watchdog
    (it would end the test process): returns the arm calls."""
    from vllm.models.deepseek_v32.amd.mono import guards

    monkeypatch.setitem(guards._FAILED, "msg", None)
    armed: list = []
    monkeypatch.setattr(guards, "_arm_exit_watchdog", lambda *a: armed.append(a))
    return armed


@pytest.fixture(autouse=True)
def mono_env(monkeypatch):
    """The mono env vars unset; no op left registered for the custom op."""
    from vllm.models.common.mono import op as mono_op

    for name in (E.ENABLE, E.CONFIG, E.FAILSTOP):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mono_op, "_ACTIVE", {})


@pytest.fixture
def no_platform_check(monkeypatch):
    """The spec's ROCm probe passes: these are CPU tests of everything around it."""
    from vllm.models.common.mono import MonoSpec

    monkeypatch.setattr(MonoSpec, "_platform", lambda self: [])


@pytest.fixture
def ckpt(tmp_path):
    """A local checkpoint dir holding the safetensors index."""
    path = tmp_path / "local_ckpt"
    path.mkdir()
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    return str(path)


@pytest.fixture
def make_vc(ckpt):
    """A VllmConfig stand-in the MonoKernel accepts; keyword args override it.
    ``graphs``: None (no compilation_config), "full" or a CUDAGraphMode."""

    def make(
        model=None,
        model_type="glm_moe_dsa",
        tp=8,
        pp=1,
        ep=False,
        spec=None,
        kv="auto",
        mml=4096,
        graphs=None,
        sizes=(1, 2, 4, 5, 6, 8),
        load_format="auto",
        download_dir=None,
    ):
        vc = NS(
            model_config=NS(
                hf_config=NS(model_type=model_type),
                model=model or ckpt,
                revision=None,
                max_model_len=mml,
                dtype=torch.bfloat16,
            ),
            parallel_config=NS(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                data_parallel_size=1,
                enable_expert_parallel=ep,
                decode_context_parallel_size=1,
            ),
            speculative_config=spec,
            cache_config=NS(cache_dtype=kv),
            load_config=NS(download_dir=download_dir, load_format=load_format),
            lora_config=None,
            kv_transfer_config=None,
            aux_output_config=NS(enable_return_routed_experts=False),
        )
        if graphs is not None:
            mode = NS(has_full_cudagraphs=lambda: True) if graphs == "full" else graphs
            vc.compilation_config = NS(
                cudagraph_mode=mode, cudagraph_capture_sizes=list(sizes)
            )
        return vc

    return make
