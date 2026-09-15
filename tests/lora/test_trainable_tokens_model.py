# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compact PEFT token replacements with a real, locally built model."""

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

VOCAB_SIZE = 64
NUM_GENERATED_TOKENS = 3
PROMPTS = [[1, 13, 11], [1, 19, 17], [1, 7, 23]]


def _load_base(model_path: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager"
    ).eval()


def _save_model(model_path: Path, tied: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=VOCAB_SIZE,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                initializer_range=0.05,
                tie_word_embeddings=tied,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    model.half().save_pretrained(model_path)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({f"token_{i}": i for i in range(4, VOCAB_SIZE)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_path)


def _save_adapter(
    model_path: Path, adapter_path: Path, token_ids: list[int] | None, seed: int
) -> None:
    base = _load_base(model_path)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            trainable_token_indices=token_ids,
            ensure_weight_tying=bool(token_ids) and base.config.tie_word_embeddings,
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.07
                )
            elif "trainable_tokens_delta" in name:
                # Despite PEFT's name, these are absolute replacement rows.
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    model.half().save_pretrained(adapter_path, save_embedding_layers=False)


@torch.inference_mode()
def _reference(model, prompt: list[int]) -> tuple[list[int], list[torch.Tensor]]:
    tokens = list(prompt)
    logprobs = []
    for _ in range(NUM_GENERATED_TOKENS):
        logits = model(torch.tensor([tokens])).logits[0, -1].float()
        # Avoid unstable greedy comparisons between nearly tied candidates.
        top_two = logits.topk(2).values
        assert float(top_two[0] - top_two[1]) > 0.03
        logprobs.append(logits.log_softmax(dim=-1))
        tokens.append(int(logits.argmax()))
    return tokens[len(prompt) :], logprobs


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires GPU LoRA kernels"
)
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("execution", ["eager", "graph", "tp2"])
def test_trainable_tokens_match_peft_with_adapter_reuse(
    tmp_path: Path, vllm_runner, tied: bool, execution: str
):
    """Token rows and normal LoRA compose without leaking across reused slots."""
    tp_size = 2 if execution == "tp2" else 1
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    model_path = tmp_path / "base"
    _save_model(model_path, tied)
    adapter_paths = [tmp_path / f"adapter_{i}" for i in range(1, 4)]
    for path, token_ids, seed in zip(
        adapter_paths, [[11, 19], [23], None], [41, 42, 43]
    ):
        _save_adapter(model_path, path, token_ids, seed)

    base = _load_base(model_path)
    expected = {0: [_reference(base, prompt) for prompt in PROMPTS]}
    for adapter_id, path in enumerate(adapter_paths, 1):
        reference = PeftModel.from_pretrained(_load_base(model_path), path).eval()
        expected[adapter_id] = [_reference(reference, prompt) for prompt in PROMPTS]
    # These checkpoints must exercise both the token replacements and real LoRA.
    for adapter_id in (1, 2, 3):
        assert not torch.allclose(
            expected[adapter_id][0][1][0], expected[0][0][1][0], atol=0.02
        )
    del reference, base

    requests = {
        i: LoRARequest(f"adapter_{i}", i, str(path))
        for i, path in enumerate(adapter_paths, 1)
    }
    params = SamplingParams(
        temperature=0,
        max_tokens=NUM_GENERATED_TOKENS,
        ignore_eos=True,
        logprobs=VOCAB_SIZE,
    )
    with vllm_runner(
        str(model_path),
        dtype="half",
        enforce_eager=execution != "graph",
        tensor_parallel_size=tp_size,
        compilation_config=CompilationConfig(max_cudagraph_capture_size=16),
        max_model_len=64,
        max_num_seqs=9,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        max_logprobs=VOCAB_SIZE,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=2,
        max_lora_rank=8,
        max_lora_trainable_tokens=2,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        gpu_memory_utilization=0.2,
    ) as runner:
        llm = runner.get_llm()

        def check_batch(adapter_ids: list[int]) -> None:
            cases = [(adapter_id, i) for i in range(3) for adapter_id in adapter_ids]
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=PROMPTS[i]) for _, i in cases],
                params,
                lora_request=[requests.get(adapter_id) for adapter_id, _ in cases],
                use_tqdm=False,
            )
            assert len(outputs) == len(cases)
            for result, (adapter_id, prompt_index) in zip(outputs, cases):
                output = result.outputs[0]
                token_ids, reference_logprobs = expected[adapter_id][prompt_index]
                assert list(output.token_ids) == token_ids, (
                    execution,
                    tied,
                    adapter_id,
                    prompt_index,
                )
                assert output.logprobs is not None
                assert len(output.logprobs) == len(reference_logprobs)
                for actual, reference in zip(output.logprobs, reference_logprobs):
                    assert set(actual) == set(range(VOCAB_SIZE))
                    actual_values = torch.tensor(
                        [actual[i].logprob for i in range(VOCAB_SIZE)]
                    )
                    torch.testing.assert_close(
                        actual_values, reference, atol=0.015, rtol=0.003
                    )

        check_batch([0, 1, 2])
        # Loading a third adapter evicts a compact adapter from the two-slot LRU.
        check_batch([0, 3])
        check_batch([0, 1, 2])
        assert llm.llm_engine.remove_lora(1)
        # Ordinary LoRA must clear any selected-token rows in its reused slot.
        check_batch([0, 2, 3])
        check_batch([0, 1, 2])
