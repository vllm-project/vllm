# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swap draft weights in a running engine: outputs stay those of the target
(verification is lossless) while the acceptance length tracks the draft quality."""

import shutil
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from vllm import LLM, SamplingParams

from ..utils import get_test_prompts


def _spec_decode_counters(llm: LLM) -> dict[str, float]:
    counters: dict[str, float] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name and hasattr(metric, "value"):
            counters[metric.name] = counters.get(metric.name, 0.0) + metric.value
    return counters


def _mean_acceptance_length(llm: LLM, prompts, sampling_params) -> tuple[float, list]:
    before = _spec_decode_counters(llm)
    outputs = llm.chat(prompts, sampling_params)
    after = _spec_decode_counters(llm)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = sum(v for k, v in delta.items() if k.endswith("spec_decode_num_drafts"))
    accepted = sum(
        v for k, v in delta.items() if k.endswith("spec_decode_num_accepted_tokens")
    )
    assert drafts > 0
    return 1.0 + accepted / drafts, [o.outputs[0].token_ids for o in outputs]


def _degraded_draft_checkpoint(draft_model: str, out_dir: Path) -> Path:
    """Copy the draft checkpoint with its lm_head zeroed (the drafter then proposes
    the same token everywhere, so almost nothing is accepted)."""
    src = Path(snapshot_download(draft_model))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, out_dir / name)
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    for f in sorted(src.glob("*.bin")):
        tensors.update(torch.load(str(f), map_location="cpu", weights_only=True))
    head_keys = [k for k in tensors if k.endswith("lm_head.weight")]
    assert head_keys, f"no lm_head in {draft_model}: {list(tensors)[:5]}"
    for k in head_keys:
        tensors[k] = torch.zeros_like(tensors[k])
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(out_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return out_dir


@pytest.mark.parametrize(
    ("model_name", "draft_model"),
    [
        ("Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3"),
        ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    ],
    ids=["qwen3_eagle3", "llama3_eagle3"],
)
def test_reload_draft_weights(tmp_path: Path, model_name: str, draft_model: str):
    prompts = get_test_prompts(mm_enabled=False)[:16]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    llm = LLM(
        model=model_name,
        speculative_config={
            "method": "eagle3",
            "model": draft_model,
            "num_speculative_tokens": 3,
            "max_model_len": 2048,
        },
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
    )
    acc_original, ref_outputs = _mean_acceptance_length(llm, prompts, sampling_params)
    assert acc_original > 1.5

    degraded = _degraded_draft_checkpoint(draft_model, tmp_path / "degraded")
    llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": str(degraded)})
    acc_degraded, outputs = _mean_acceptance_length(llm, prompts, sampling_params)
    assert outputs == ref_outputs, "reloading the draft changed target outputs"
    assert acc_degraded < acc_original - 0.5

    llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": draft_model})
    acc_restored, outputs = _mean_acceptance_length(llm, prompts, sampling_params)
    assert outputs == ref_outputs
    assert abs(acc_restored - acc_original) < 0.1
