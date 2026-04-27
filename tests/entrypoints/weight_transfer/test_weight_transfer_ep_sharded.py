# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for EP-sharded weight updates wiring.

These tests spin up a tiny real model and verify:

1. ``GPUWorker.get_moe_routed_ep_layout`` RPC returns the expected per-layer
   EP layout metadata used by mbridge.
2. ``FusedMoE.load_routed_expert_weights`` writes fused 3D
   ``[local_routed_E, 2I, H]`` input tensors into live model parameters
   bit-exactly when invoked through the model's live FusedMoE layers.
3. A regression: when ``moe_routed_expert_global_ids`` is None, the
   ``update_weights`` callback delegates to the model's regular
   ``load_weights`` path (no routing to sharded loader).

The tiny model used is Qwen3Next (unfused checkpoint format). For the
sharded-load test we explicitly inject a **fused** ``expert_mapping`` into
the live FusedMoE layer so we can exercise the sharded path without needing
a fused-checkpoint tiny model (which is not readily available).
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm import LLM
from vllm.config import WeightTransferConfig

from ...utils import create_new_process_for_each_test

MOE_MODEL_NAME = "tiny-random/qwen3-next-moe"


def _prep_env():
    # Run in-process so patches/state in workers are visible.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU.",
)
@create_new_process_for_each_test()
def test_get_moe_routed_ep_layout_returns_expected_shape():
    """``GPUWorker.get_moe_routed_ep_layout`` must return the per-layer EP
    layout dict used by mbridge to plan sharded sends."""
    _prep_env()

    llm = LLM(
        model=MOE_MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    def collect_layouts(self):
        return self.get_moe_routed_ep_layout()

    per_rank = llm.collective_rpc(collect_layouts)
    assert len(per_rank) == 1
    layout = per_rank[0]
    assert isinstance(layout, dict) and len(layout) > 0

    sample = next(iter(layout.values()))
    required = {
        "ep_rank",
        "ep_size",
        "local_num_routed_experts",
        "global_num_routed_experts",
        "num_fused_shared_experts",
        "local_to_global_routed",
    }
    assert required.issubset(sample.keys())
    assert len(sample["local_to_global_routed"]) == sample["local_num_routed_experts"]
    assert all(
        0 <= g < sample["global_num_routed_experts"]
        for g in sample["local_to_global_routed"]
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU.",
)
@create_new_process_for_each_test()
def test_load_routed_expert_weights_on_live_fused_moe():
    """Inject a fused ``expert_mapping`` into the live FusedMoE layer and
    invoke ``load_routed_expert_weights`` with a fused 3D tensor. Verify
    bit-exact writes into the stacked ``w13_weight`` / ``w2_weight`` params."""
    _prep_env()

    llm = LLM(
        model=MOE_MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    FUSED_MAPPING = [
        ("w13_weight", "gate_up_proj", 0, "w1"),
        ("w13_weight", "gate_up_proj", 1, "w3"),
        ("w2_weight", "down_proj", 0, "w2"),
    ]

    def exercise_sharded_load(self):
        """Runs inside the worker; returns a dict summarizing the outcome."""
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        model = self.model_runner.model
        first = next((m for m in model.modules() if isinstance(m, FusedMoE)), None)
        if first is None:
            return {"ok": False, "reason": "no FusedMoE layer"}

        # Inject fused mapping for this test only.
        first.expert_mapping = FUSED_MAPPING

        local_to_global = (
            (first._expert_map != -1).nonzero().flatten().tolist()
            if first._expert_map is not None
            else list(range(first.global_num_experts))
        )
        local_E = len(local_to_global)

        w13 = first.w13_weight.data
        w2 = first.w2_weight.data
        device = w13.device
        dtype = w13.dtype

        # Build fused 3D source tensors with per-expert unique patterns.
        w13_src = torch.zeros(
            local_E, w13.shape[1], w13.shape[2], dtype=dtype, device=device
        )
        w2_src = torch.zeros(
            local_E, w2.shape[1], w2.shape[2], dtype=dtype, device=device
        )
        for i, gid in enumerate(local_to_global):
            w13_src[i].fill_(float(gid + 1))
            w2_src[i].fill_(-float(gid + 1))

        # Zero out current params so post-write comparison is unambiguous.
        w13.zero_()
        w2.zero_()

        # expert_name is the suffix that replaces the mapping's weight_name
        # inside qual_name = f"{layer_name}.{expert_name}". With the flat
        # mapping below, that resolves to the flat FusedMoE attr w13_weight /
        # w2_weight directly.
        list(
            first.load_routed_expert_weights(
                [("gate_up_proj", w13_src)],
                {"gate_up_proj": local_to_global},
            )
        )
        list(
            first.load_routed_expert_weights(
                [("down_proj", w2_src)],
                {"down_proj": local_to_global},
            )
        )

        # Verify bit-exact writes.
        ok = True
        mismatches: list[str] = []
        for i, gid in enumerate(local_to_global):
            if not torch.all(w13[i] == float(gid + 1)).item():
                ok = False
                mismatches.append(
                    f"w13 local={i} gid={gid} sample={w13[i].flatten()[:3].tolist()}"
                )
            if not torch.all(w2[i] == -float(gid + 1)).item():
                ok = False
                mismatches.append(
                    f"w2 local={i} gid={gid} sample={w2[i].flatten()[:3].tolist()}"
                )
        return {
            "ok": ok,
            "layer": first.layer_name,
            "local_E": local_E,
            "local_to_global": local_to_global,
            "mismatches": mismatches[:5],
        }

    results = llm.collective_rpc(exercise_sharded_load)
    for r in results:
        assert r["ok"], f"sharded write failed: {r}"


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU.",
)
@create_new_process_for_each_test()
def test_update_weights_without_ids_map_regression():
    """Backward-compat: when ``moe_routed_expert_global_ids`` is absent, the
    callback must delegate to ``model.load_weights`` unchanged — verified by
    updating a real model parameter via the full ``LLM.update_weights`` API
    path and confirming the value lands correctly."""
    _prep_env()

    from collections.abc import Callable
    from dataclasses import dataclass
    from unittest.mock import patch

    from vllm.distributed.weight_transfer.base import (
        WeightTransferEngine,
        WeightTransferInitInfo,
        WeightTransferInitRequest,
        WeightTransferUpdateInfo,
        WeightTransferUpdateRequest,
    )

    @dataclass
    class _InitInfo(WeightTransferInitInfo):
        pass

    @dataclass
    class _UpdateInfo(WeightTransferUpdateInfo):
        names: list[str] | None = None
        dtype_names: list[str] | None = None
        shapes: list[list[int]] | None = None

    class _SynthEngine(WeightTransferEngine[_InitInfo, _UpdateInfo]):
        """receive_weights fills zero tensors from update_info and invokes
        load_weights — equivalent to pretending an IPC transfer arrived."""

        init_info_cls = _InitInfo
        update_info_cls = _UpdateInfo

        def __init__(self, config, parallel_config):
            super().__init__(config, parallel_config)

        def init_transfer_engine(self, init_info):
            pass

        def shutdown(self):
            pass

        @staticmethod
        def trainer_send_weights(*args, **kwargs):
            pass

        def receive_weights(
            self,
            update_info,
            load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
        ):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dev = torch.accelerator.current_device_index()
            weights = []
            for name, dstr, shape in zip(
                update_info.names or [],
                update_info.dtype_names or [],
                update_info.shapes or [],
            ):
                t = torch.zeros(shape, dtype=dtype_map[dstr], device=f"cuda:{dev}")
                weights.append((name, t))
            load_weights(weights)

    def _factory(config, parallel_config):
        return _SynthEngine(config, parallel_config)

    with patch(
        "vllm.v1.worker.gpu_worker.WeightTransferEngineFactory.create_engine",
        _factory,
    ):
        llm = LLM(
            model=MOE_MODEL_NAME,
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.03,
            max_model_len=1024,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )
        llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info={}))

        # Pick an existing simple 1D param — model.norm.weight — that
        # model.load_weights can write without any remapping. Pre-seed it to
        # a non-zero sentinel, then trigger update (which zeroes it via the
        # synth engine), and verify the zero lands.
        def probe(self):
            model = self.model_runner.model
            params = dict(model.named_parameters())
            # Pick a leaf param that is 1D and has a simple weight_loader.
            candidate = None
            for name, p in params.items():
                if name.endswith("norm.weight") and p.dim() == 1:
                    candidate = (name, list(p.shape), str(p.dtype).split(".")[-1])
                    break
            return candidate

        cand = llm.collective_rpc(probe)[0]
        assert cand is not None, "No norm.weight-like param found"
        name, shape, dtype_name = cand

        def seed(self):
            p = self.model_runner.model.get_parameter(name)
            p.data.fill_(42.0)
            return float(p.data.flatten()[0].item())

        assert all(v == 42.0 for v in llm.collective_rpc(seed))

        llm.update_weights(
            WeightTransferUpdateRequest(
                update_info={
                    "names": [name],
                    "dtype_names": [dtype_name],
                    "shapes": [shape],
                }
            )
        )

        def verify(self):
            p = self.model_runner.model.get_parameter(name)
            return bool(torch.all(p.data == 0).item())

        assert all(llm.collective_rpc(verify)), (
            f"param {name} was not overwritten by update_weights callback"
        )
