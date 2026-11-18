# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA and generation regressions for last-layer FFN pruning."""

from typing import TYPE_CHECKING

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.v1.worker.gpu.last_layer_ffn import (
    LastLayerFFNGraphWrapper,
    last_layer_ffn_graph_key,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_first_capture_executes_and_replay_uses_current_rows(dist_init, monkeypatch):
    """New row-count graphs must execute immediately and reuse live indices."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = VllmConfig()

    def select(x, last_layer_token_indices):
        return x.index_select(0, last_layer_token_indices) + 1

    graph = LastLayerFFNGraphWrapper(select, config, key_fn=last_layer_ffn_graph_key)
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    indices = torch.tensor([0, 2], device="cuda")
    with set_forward_context(
        None,
        config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=4),
    ):
        for rows, offset, selected_rows in [
            (1, 0, [0, 2]),
            (2, 16, [1, 3]),
            (1, 32, [3, 0]),
            (2, 48, [2, 1]),
        ]:
            x.copy_(torch.arange(16, device="cuda").reshape(4, 4) + offset)
            indices.copy_(torch.tensor(selected_rows, device="cuda"))
            selected = indices[:rows]
            actual = graph(x, last_layer_token_indices=selected)
            torch.testing.assert_close(actual, select(x, selected), rtol=0, atol=0)
    assert len(graph.entries) == 2


class FFNTestWorkerExtension:
    model_runner: "GPUModelRunner"

    def ffn_test_stats(self):
        runner = self.model_runner
        pruner = runner.last_layer_ffn_pruner
        manager = runner.cudagraph_manager
        graph = manager.breakable_cg_runner if manager is not None else None
        return {
            "batches": pruner.batches,
            "rows": pruner.skipped_rows,
            "graphs": len(graph.entries) if graph is not None else 0,
        }


def _stats(model):
    return model.llm.collective_rpc("ffn_test_stats")[0]


@pytest.mark.parametrize(
    "model_name",
    ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"],
)
@pytest.mark.parametrize("mode", ["eager", "graph", "prefix", "chunked"])
def test_pruning_preserves_generation_and_fallbacks(
    vllm_runner, monkeypatch, mode, model_name
):
    """Exercise capture/replay and unsupported prefills through real weights."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0" if mode == "eager" else "1")
    chunked = mode == "chunked"
    prefix = mode == "prefix"
    prompts = [
        "The following is a short arithmetic exercise. " * 4 + "7 times 8 equals",
        "The following is a short geography exercise. " * 4
        + "The capital of France is",
    ]
    if chunked:
        prompts = [p * 4 for p in prompts]
    outputs = []
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_ENABLE_LAST_LAYER_FFN_PRUNING", str(int(enabled)))
        with vllm_runner(
            maybe_model_redirect(model_name),
            dtype="bfloat16",
            max_model_len=1024,
            max_num_seqs=2,
            max_num_batched_tokens=64 if chunked else 1024,
            gpu_memory_utilization=0.8,
            enforce_eager=mode == "eager",
            enable_prefix_caching=prefix,
            enable_chunked_prefill=chunked,
            attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "prefill_cudagraph_capture_token_sizes": [32, 64]
                if chunked
                else [128, 256],
                "decode_cudagraph_capture_request_sizes": [1, 2],
            },
            worker_extension_cls=f"{__name__}.FFNTestWorkerExtension",
        ) as model:
            initial = _stats(model)
            arm = []
            # Revisit the one-request variant after capturing the two-request one.
            for batch in ([prompts[0]], prompts, [prompts[1]]):
                arm.append(model.generate_greedy_logprobs(batch, 8, 5))
            after = _stats(model)
            if enabled and not chunked:
                assert after["batches"] > initial["batches"]
                assert after["rows"] > initial["rows"]
            else:
                assert after["batches"] == initial["batches"]
            if mode != "eager":
                assert after["graphs"] > 0
            if prefix:
                result = model.llm.generate(
                    prompts,
                    SamplingParams(temperature=0, max_tokens=1),
                    use_tqdm=False,
                )
                assert all(r.num_cached_tokens > 0 for r in result)
                assert _stats(model)["batches"] == after["batches"]
            # Prompt scoring consumes earlier rows, so pruning must be bypassed.
            before_scoring = _stats(model)
            arm.append(
                model.generate_greedy_logprobs(
                    ["A new prompt for scoring: the first prime numbers are"],
                    8,
                    5,
                    num_prompt_logprobs=1,
                )
            )
            assert _stats(model)["batches"] == before_scoring["batches"]
            outputs.append(arm)
    for stock, pruned in zip(*outputs):
        check_logprobs_close(
            outputs_0_lst=stock,
            outputs_1_lst=pruned,
            name_0="stock",
            name_1="last-layer-ffn",
        )
