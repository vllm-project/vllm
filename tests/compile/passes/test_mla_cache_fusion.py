# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from uuid import uuid4

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

FUSED_OP = "torch.ops.vllm.fused_rope_and_unified_mla_kv_cache_update.default"


def _count_in_rope_fusion_after_dump(dump_dir: Path, needle: str) -> int:
    graph_files = list(dump_dir.rglob("__compiled_fn_*.kernel_*.py"))
    if not graph_files:
        graph_files = list(dump_dir.rglob("__compiled_fn_*.py"))

    total = 0
    for path in graph_files:
        total += path.read_text(errors="ignore").count(needle)
    return total


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not torch.cuda.is_available(),
    reason="MLA cache fusion test requires a CUDA-alike GPU device.",
)
def test_mla_cache_fusion_with_deepseek_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-level check that MLA RoPE+cache update fusion is applied."""

    model = "deepseek-ai/DeepSeek-V2-Lite"
    dump_dir = tmp_path / f"debug_dump_{uuid4().hex}"

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_DEBUG_DUMP_PATH", str(dump_dir))

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        max_model_len=1024,
        compilation_config={
            "use_inductor_graph_partition": True,
            "pass_config": {
                "fuse_rope_kvcache": True,
            },
            "cudagraph_mode": "NONE",
        },
        load_format="dummy",
    )

    outputs = llm.generate(
        ["Hello", "Give me one short fact about the moon."],
        SamplingParams(temperature=0.0, max_tokens=3),
    )

    assert len(outputs) == 2
    assert all(len(o.outputs[0].text) > 0 for o in outputs)

    fused_count = _count_in_rope_fusion_after_dump(dump_dir, FUSED_OP)

    assert fused_count == 27, (
        "Expected exactly 27 fused_rope_and_unified_mla_kv_cache_update "
        "hits for DeepSeek-V2-Lite."
    )
