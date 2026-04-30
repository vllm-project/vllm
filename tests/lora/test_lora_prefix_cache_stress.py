# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


def _repeat_to_length(token_ids: list[int], target_len: int) -> list[int]:
    return (token_ids * ((target_len + len(token_ids) - 1) // len(token_ids)))[
        :target_len
    ]


def _build_prompt_token_ids(
    tokenizer, request_id: str, prompt_len: int, prefix_len: int
) -> list[int]:
    shared_unit = tokenizer.encode(
        "Context: the stable key is blue, the checksum is seven, "
        "and the requested output is the word seven.\n",
        add_special_tokens=False,
    )
    filler_unit = tokenizer.encode(
        f"Record {request_id}: neutral filler, no new facts. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nQuestion: What single word should be returned?\nAnswer: seven",
        add_special_tokens=False,
    )
    assert prompt_len >= prefix_len + len(suffix)
    prefix = _repeat_to_length(shared_unit, prefix_len)
    filler = _repeat_to_length(
        filler_unit, prompt_len - prefix_len - len(suffix)
    )
    return prefix + filler + suffix


def _generate_random_lora(output_dir: str, seed: int) -> str:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)
    peft_model.save_pretrained(output_dir)
    del peft_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


@pytest.mark.optional
def test_lora_alternation_prefix_cache_runtime_stress(tmp_path):
    """Generated LoRA adapters stay deterministic under chunked prefix reuse.

    This mirrors the issue #38606 traffic shape, but keeps the assertion
    simple: repeated greedy runs of the same alternating-adapter batch must
    produce identical text. The core block lifecycle invariant is covered by
    tests/v1/core/test_prefix_caching.py.
    """
    lora_a_path = _generate_random_lora(str(tmp_path / "lora_a"), seed=1)
    lora_b_path = _generate_random_lora(str(tmp_path / "lora_b"), seed=2)
    lora_a = LoRARequest("lora_a", 1, lora_a_path)
    lora_b = LoRARequest("lora_b", 2, lora_b_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_tokens=8,
    )
    requests = [
        ("r1", lora_a, 700, 512),
        ("thrash_0", lora_a, 700, 512),
        ("thrash_1", lora_b, 700, 0),
        ("thrash_2", lora_a, 700, 0),
        ("thrash_3", lora_b, 700, 0),
        ("thrash_4", lora_a, 700, 0),
        ("thrash_5", lora_b, 700, 0),
        ("thrash_6", lora_a, 700, 0),
        ("r2", lora_b, 700, 512),
    ]
    prompts = [
        {
            "prompt_token_ids": _build_prompt_token_ids(
                tokenizer, request_id, prompt_len, prefix_len
            )
        }
        for request_id, _, prompt_len, prefix_len in requests
    ]
    lora_requests = [lora for _, lora, _, _ in requests]

    def run_once() -> list[str]:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests,
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]

    # Warm the cache once so the assertion compares repeated prefix-cache
    # reuse, not uncached prefill against cached prefill. Those two paths can
    # differ by bf16 tie-breaking without implying KV block corruption.
    run_once()
    baseline = run_once()
    for _ in range(3):
        assert run_once() == baseline
