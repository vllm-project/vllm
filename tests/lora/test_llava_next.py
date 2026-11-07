# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .utils import create_peft_lora

MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
TARGETS = [
    "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
    "multi_modal_projector.linear_1",
]


def _write_adapters(model: torch.nn.Module, paths: list[str]) -> None:
    with torch.random.fork_rng(devices=[]):
        for seed, path in enumerate(paths):
            torch.manual_seed(seed)
            weights = create_peft_lora(model, path, TARGETS, rank=8, alpha=8, dropout=0)
            for name, weight in weights.items():
                if name.endswith("lora_B.weight"):
                    weight.normal_(std=0.01)
            save_file(weights, str(Path(path) / "adapter_model.safetensors"))


def test_anyres_tower_connector_lora_mixed_batch(
    vllm_runner, tmp_path, monkeypatch, enable_pickle
):
    """Unpadded image placeholders must not under-size pre-unpadding LoRA rows."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # LoRA kernels read batch invariance at import time in the worker.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    paths = [str(tmp_path / f"adapter-{i}") for i in range(2)]
    for path in paths:
        Path(path).mkdir()
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (500, 336, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    prompt = "[INST] <image>\nDescribe the image briefly. [/INST]"
    adapters = [LoRARequest(str(i), i, path) for i, path in enumerate(paths, 1)]
    routes = [adapters[0], adapters[1], None, adapters[0], adapters[1]]
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=5)

    with vllm_runner(
        MODEL,
        dtype="float16",
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        kv_cache_memory_bytes=2 * 1024**3,
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_lora_rank=8,
        max_loras=2,
    ) as runner:
        llm = runner.get_llm()
        llm.apply_model(partial(_write_adapters, paths=paths))
        individual = [
            runner.generate_w_logprobs(
                [prompt], params, images=[image], lora_request=adapter
            )[0]
            for image, adapter in zip(images, routes)
        ]
        # Re-encode all five images together rather than reusing the references.
        llm.llm_engine.reset_encoder_cache()
        # 7380 placeholders fit the budget; tower/connector require 8655/8640 rows.
        inputs = runner.get_inputs([prompt] * 5, images=images)
        llm.sleep(level=0)
        try:
            llm.enqueue(inputs, params, lora_request=routes, use_tqdm=False)
        finally:
            llm.wake_up(tags=["scheduling"])
        mixed = llm.wait_for_completion(RequestOutput, use_tqdm=False)

    assert len(mixed) == len(individual) == 5
    for request_index, (reference, output) in enumerate(zip(individual, mixed)):
        ref_tokens, _, ref_logprobs = reference
        actual = output.outputs[0]
        tokens, logprobs = list(actual.token_ids), actual.logprobs
        assert logprobs is not None and ref_logprobs is not None
        assert tokens == ref_tokens, f"request {request_index}: mixed tokens differ"
        assert len(tokens) == len(logprobs) == len(ref_logprobs) == 4
        max_error = 0.0
        for position, (expected, observed) in enumerate(zip(ref_logprobs, logprobs)):
            context = f"request {request_index}, token {position}"
            assert observed.keys() == expected.keys(), context
            token_ids = sorted(expected)
            ref_values = torch.tensor([expected[t].logprob for t in token_ids])
            values = torch.tensor([observed[t].logprob for t in token_ids])
            max_error = max(max_error, (values - ref_values).abs().max().item())
            # 0.002 log-prob error is approximately 0.2% probability error.
            torch.testing.assert_close(
                values, ref_values, rtol=0, atol=2e-3, msg=context
            )
        print(
            f"LLaVA-NeXT mixed vs individual logprobs: request={request_index}, "
            f"tokens={tokens}, max_abs_error={max_error:.8g}"
        )
