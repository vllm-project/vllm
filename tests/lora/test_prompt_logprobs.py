# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform


@pytest.fixture(scope="module")
def prompt_scoring_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("prompt_scoring_lora")
    hidden_size, vocab_size, rank = 64, 128, 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=hidden_size,
                intermediate_size=128,
                num_attention_heads=1,
                num_hidden_layers=1,
                vocab_size=vocab_size,
            )
        )
        model.save_pretrained(root / "model")
    generator = torch.Generator().manual_seed(0)
    layer = "base_model.model.model.layers.0.self_attn.v_proj"
    weights = {
        f"{layer}.lora_A.weight": torch.randn(rank, hidden_size, generator=generator),
        f"{layer}.lora_B.weight": torch.randn(hidden_size, rank, generator=generator),
    }
    head_b = torch.zeros(vocab_size, rank)
    head_b[7, 0] = 64
    head_b[8, 0] = -64
    head_weights = {
        "base_model.model.lm_head.lora_A.weight": torch.randn(
            rank, hidden_size, generator=generator
        ),
        "base_model.model.lm_head.lora_B.weight": head_b,
    }
    for name, adapter_weights in (
        ("body", weights),
        ("head", head_weights),
    ):
        adapter_dir = root / name
        adapter_dir.mkdir()
        # Both configs list lm_head. Only the head checkpoint has its tensors.
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {"r": rank, "lora_alpha": rank, "target_modules": ["v_proj", "lm_head"]}
            )
        )
        save_file(adapter_weights, adapter_dir / "adapter_model.safetensors")
    return str(root / "model"), str(root / "body"), str(root / "head")


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU LoRA regression")
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_lora_prompt_scoring(
    vllm_runner, prompt_scoring_model, enforce_eager, monkeypatch
):
    """Scoring uses the body adapter and rejects only requests with head weights."""
    if enforce_eager:
        # Submit the full mixed batch before the in-process engine can step it.
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_path, body_path, head_path = prompt_scoring_model
    body = LoRARequest("body", 1, body_path)
    head = LoRARequest("head", 2, head_path)
    short_ids = [11, 22, 33]
    short_prompt = {"prompt_token_ids": short_ids}
    long_ids = [11, 22, 33, 44] * 256 + [55, 66, 77]
    long_prompt = {"prompt_token_ids": long_ids}
    # The mixed batch has 2 * 1024 + 1 token rows and four sampler rows.
    base_batch_ids = [11, 22] * 508
    generation = SamplingParams(
        temperature=0, max_tokens=2, ignore_eos=True, detokenize=False
    )
    standard = SamplingParams(
        temperature=0, max_tokens=1, prompt_logprobs=0, detokenize=False
    )
    fixed_ids_only = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprob_token_ids=[[22], [33]],
        detokenize=False,
    )
    both = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=0,
        prompt_logprob_token_ids=[[token_id] for token_id in long_ids[1:]],
        detokenize=False,
    )
    with vllm_runner(
        model_path,
        enforce_eager=enforce_eager,
        dtype="float16",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=4,
        max_num_batched_tokens=2112,
        max_model_len=1152,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=64 << 20,
        enable_flashinfer_autotune=False,
        compilation_config={
            "decode_cudagraph_capture_request_sizes": [1, 2, 4],
            "prefill_cudagraph_capture_token_sizes": [4],
        },
        skip_tokenizer_init=True,
    ) as runner:
        llm = runner.llm
        assert llm.llm_engine.add_lora(head)
        for scoring in (standard, fixed_ids_only):
            rejected = llm.generate(short_prompt, scoring, lora_request=head)[0]
            completion = rejected.outputs[0]
            assert completion.finish_reason == "error"
            assert "lm_head" in completion.stop_reason
            assert not completion.token_ids
            assert rejected.prompt_token_id_logprobs is None

        base = llm.generate(long_prompt, standard)[0]
        base_generation = llm.generate(short_prompt, generation)[0]
        llm.enqueue(long_prompt, both, lora_request=body)
        llm.enqueue(short_prompt, generation, lora_request=head)
        llm.enqueue({"prompt_token_ids": base_batch_ids}, standard)
        llm.enqueue(short_prompt, standard, lora_request=head)
        adapted, head_generation, base_in_batch, rejected_in_batch = (
            llm.wait_for_completion()
        )

        assert rejected_in_batch.outputs[0].finish_reason == "error"
        assert len(head_generation.outputs[0].token_ids) == 2
        assert set(head_generation.outputs[0].token_ids) <= {7, 8}
        assert base_generation.outputs[0].token_ids[0] not in (7, 8)
        assert len(base_in_batch.prompt_logprobs) == len(base_batch_ids)
        assert len(base.prompt_logprobs) == len(long_ids)
        assert len(adapted.prompt_logprobs) == len(long_ids)
        base_scores, adapted_scores = (
            np.array(
                [
                    scores[token_id].logprob
                    for token_id, scores in zip(long_ids[1:], out.prompt_logprobs[1:])
                ]
            )
            for out in (base, adapted)
        )
        assert np.max(np.abs(base_scores - adapted_scores)) > 0.1
        np.testing.assert_allclose(
            adapted.prompt_token_id_logprobs[:, 0], adapted_scores, atol=1e-3
        )

        assert llm.llm_engine.remove_lora(head.lora_int_id)
        replacement = LoRARequest("replacement", head.lora_int_id, body_path)
        assert llm.llm_engine.add_lora(replacement)
        (replaced,) = llm.generate(
            short_prompt, fixed_ids_only, lora_request=replacement
        )
        assert replaced.outputs[0].finish_reason == "length"
        assert np.isfinite(replaced.prompt_token_id_logprobs).all()

        if enforce_eager:
            llm.enqueue(short_prompt, standard, lora_request=replacement)
            assert llm.llm_engine.add_lora(
                LoRARequest(
                    "replacement", head.lora_int_id, head_path, load_inplace=True
                )
            )
            (reloaded,) = llm.wait_for_completion()
            assert reloaded.outputs[0].finish_reason == "error"
