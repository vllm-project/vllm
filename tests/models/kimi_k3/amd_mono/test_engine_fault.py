# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-stop end to end: a TP rank that stalls inside a served Kimi-K3
MonoKernel step fails that step, and the engine stops instead of hanging or
returning the step's tokens."""

import json
import os
import time

import pytest
import requests

from tests.models.kimi_k3.amd_mono.test_numerics import needs_gfx950
from tests.utils import RemoteOpenAIServer, multi_gpu_test

MODEL = "riverclouds/Kimi-K3-8L-dummy"
DRAFT = "riverclouds/Kimi-K3-DSpark-for-8L-dummy"
STALLED = 3
FAULT_EPOCH = 256  # a mono decode step well past warm-up and capture
STARTUP_TIMEOUT = 1800
FAULT_DEADLINE = 900
EXIT_TIMEOUT = 300

# Builds every rank's K2 with the fault hook (``K2Build.fault_*``): CTA 0 of
# rank STALLED stalls past every in-kernel wait's bound in the step at
# FAULT_EPOCH. Patched into ``MONO_SPEC.pack`` as the spec module loads, in
# every vLLM process.
_SITECUSTOMIZE = """\
import dataclasses
import importlib.abc
import importlib.machinery
import sys

_MODULE = "vllm.models.kimi_k3.amd.mono.spec"


def _patch(module):
    pack = module.MONO_SPEC.pack

    def faulty_pack(model, ctx):
        runner = pack(model, ctx)
        runner.fault = {{
            "fault_rank": {rank},
            "fault_epoch": {epoch},
            "fault_layer": 128 + runner.first_layer,  # the step's first K2
            "fault_ticks": {ticks},
        }}
        return runner

    module.MONO_SPEC = dataclasses.replace(module.MONO_SPEC, pack=faulty_pack)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = patched_exec
        return spec


sys.meta_path.insert(0, _Finder())
"""

SERVER_ARGS = [
    "--trust-remote-code",
    "--language-model-only",
    "--load-format",
    "dummy",
    "--tensor-parallel-size",
    "8",
    "--decode-context-parallel-size",
    "8",
    "--dcp-comm-backend",
    "a2a",
    "--kv-cache-dtype",
    "fp8",
    "--max-model-len",
    "8192",
    "--max-num-seqs",
    "2",
    "--attention-config",
    json.dumps({"mla_prefill_backend": "ROCM_AITER_FA"}),
    "--speculative-config",
    json.dumps(
        {
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": 7,
            "attention_backend": "ROCM_AITER_MLA",
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        }
    ),
    "--compilation-config",
    json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    "--monokernel",
]


def _mono_steps(url: str) -> float:
    text = requests.get(f"{url}/metrics", timeout=30).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("vllm:monokernel_steps_total{") and 'path="mono"' in line
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_a_stalled_rank_stops_the_engine(tmp_path):
    from vllm.models.kimi_k3.amd.ops.mono.common.sync import WAIT_TICKS

    site = tmp_path / "mono_fault"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        _SITECUSTOMIZE.format(
            rank=STALLED, epoch=FAULT_EPOCH, ticks=WAIT_TICKS * 3 // 2
        )
    )
    env = {
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(site), os.getenv("PYTHONPATH")])
        ),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_AITER_MLA_ASM_PADDING": "asm",
    }
    body = {
        "model": MODEL,
        "prompt": "hello " * 64,
        "max_tokens": 128,
        "temperature": 0,
        "ignore_eos": True,
    }
    with RemoteOpenAIServer(
        MODEL, SERVER_ARGS, env_dict=env, max_wait_seconds=STARTUP_TIMEOUT
    ) as server:
        url = server.url_root
        served = 0
        deadline = time.monotonic() + FAULT_DEADLINE
        while time.monotonic() < deadline:
            try:
                r = requests.post(f"{url}/v1/completions", json=body, timeout=300)
            except requests.ConnectionError:
                break
            if r.status_code != 200:
                break
            if served == 0:
                assert _mono_steps(url) > 0, "decode never ran on the MonoKernel"
            served += 1
        else:
            pytest.fail(f"no request failed in {FAULT_DEADLINE}s: the fault is lost")

        assert served > 0, "the engine failed before the injected step"
        # the failed step stops the engine and the server exits
        assert server.proc.wait(timeout=EXIT_TIMEOUT) is not None
