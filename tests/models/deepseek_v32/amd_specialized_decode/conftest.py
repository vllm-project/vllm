# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures of the GLM-5.2 specialized decode (ROCm, specialized_decode/) CPU
tests. MonoLayers and SpecializedDecode are built bare (``__new__``) around fake kernel
ops (no GPU, no FlyDSL)."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def no_device_sync(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)


@pytest.fixture
def import_kernel():
    """Import a FlyDSL kernel module, or skip the test: the kernel build needs FlyDSL
    and aiter (whose import needs an active Triton GPU driver)."""

    def load(name: str):
        try:
            return importlib.import_module(
                f"vllm.models.deepseek_v32.amd.specialized_decode.flydsl.{name}"
            )
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"FlyDSL kernel build unavailable: {e!r}")

    return load


@pytest.fixture(autouse=True)
def failstop_latch(monkeypatch):
    """No latched output-rank fail-stop, and a fail-stop does not arm the exit watchdog
    (it would end the test process): returns the arm calls."""
    from vllm.models.deepseek_v32.amd.specialized_decode.runtime import guards

    monkeypatch.setitem(guards._FAILED, "msg", None)
    armed: list = []
    monkeypatch.setattr(guards, "_arm_exit_watchdog", lambda *a: armed.append(a))
    return armed


@pytest.fixture
def make_vc():
    """A VllmConfig stand-in the specialized decode accepts (switch on); keyword args
    override it. ``graphs``: None (no cudagraph mode), "full" or a CUDAGraphMode."""

    def make(
        model_type="glm_moe_dsa",
        tp=8,
        pp=1,
        ep=False,
        spec=None,
        kv="auto",
        mml=4096,
        graphs=None,
        backend="latency_specialized",
        options=None,
    ):
        vc = NS(
            model_config=NS(
                hf_config=NS(model_type=model_type),
                max_model_len=mml,
                dtype=torch.bfloat16,
                enable_sleep_mode=False,
            ),
            parallel_config=NS(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                data_parallel_size=1,
                enable_expert_parallel=ep,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
                use_ubatching=False,
            ),
            speculative_config=spec,
            cache_config=NS(cache_dtype=kv),
            lora_config=None,
            kv_transfer_config=None,
            aux_output_config=NS(enable_return_routed_experts=False),
            additional_config={} if backend is None else {"decode_backend": backend},
            use_v2_model_runner=True,
        )
        mode = NS(has_full_cudagraphs=lambda: True) if graphs == "full" else graphs
        vc.compilation_config = NS(mode=0, cudagraph_mode=mode)
        if options is not None:
            vc.additional_config["specialized_decode"] = options
        return vc

    return make
