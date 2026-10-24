# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3_5ForSlotDecision's block attention against the transformers block mask,
on a random four-layer model built from BLOCK_ATTN_BASE."""

import os

import pytest
import torch

from tests.models.test_qwen3_5_slot import OVERRIDES
from vllm.platforms import current_platform

BASE = os.environ.get("BLOCK_ATTN_BASE", "baselabs/elephant-1")
NUM_SCORED = 256
# bf16 vLLM against an fp32 reference: a read must be close to the block-mask
# reference and much closer to it than the causal reference is.
ATOL = 0.1
MAX_ERROR_VS_MASK_GAP = 0.2

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="block attention needs a CUDA GPU"
)


def build_tiny_checkpoint(path: str) -> None:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(BASE)
    config.text_config.update(
        {
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "full_attention_interval": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 0,
            "dtype": "float32",
        }
    )
    config.vision_config.update(
        {
            "depth": 1,
            "hidden_size": 64,
            "num_heads": 2,
            "intermediate_size": 128,
            "out_hidden_size": 256,
        }
    )
    config.dtype = "float32"
    config.modeling_overrides = OVERRIDES
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
    # Sharper weights and stronger attention values/outputs make the mask move
    # the last token's logits well above bf16 noise.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                param.normal_(0.0, 0.08)
                if "self_attn.v_proj" in name or "self_attn.o_proj" in name:
                    param.mul_(4.0)
    model.save_pretrained(path)
    AutoProcessor.from_pretrained(BASE).save_pretrained(path)


def hf_logprobs(path: str, prompts, prefix_len: int | None) -> list[torch.Tensor]:
    """Last-position logprobs over the first NUM_SCORED tokens, with the prefix
    and the rest of each prompt as two blocks (causal if `prefix_len` is None)."""
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    original = modeling_qwen3_5.create_causal_mask
    current: dict = {}

    def create_causal_mask(*args, **kwargs):
        if current.get("ids") is not None:
            kwargs["block_sequence_ids"] = current["ids"]
        return original(*args, **kwargs)

    out = []
    modeling_qwen3_5.create_causal_mask = create_causal_mask
    try:
        for prompt in prompts:
            current["ids"] = None
            if prefix_len is not None:
                positions = torch.arange(len(prompt), device="cuda")
                current["ids"] = (positions >= prefix_len).long()[None]
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([prompt], device="cuda")).logits
            out.append(torch.log_softmax(logits[0, -1].float(), -1)[:NUM_SCORED].cpu())
    finally:
        modeling_qwen3_5.create_causal_mask = original
    del model
    torch.accelerator.empty_cache()
    return out


def test_block_attention_matches_transformers(vllm_runner, tmp_path):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    checkpoint = str(tmp_path)
    build_tiny_checkpoint(checkpoint)
    g = torch.Generator().manual_seed(1)

    def tokens(n: int) -> list[int]:
        return torch.randint(1000, 100000, (n,), generator=g).tolist()

    # A long shared prefix, then three questions.
    prefix = tokens(700)
    shared = tokens(40)
    questions = {
        "q0": prefix + tokens(90),
        "q1": prefix + shared + tokens(50),
        "q2": prefix + shared + tokens(50),
    }
    p = len(prefix)
    names = list(questions)
    expected = dict(zip(names, hf_logprobs(checkpoint, questions.values(), p)))
    causal = dict(zip(names, hf_logprobs(checkpoint, questions.values(), None)))
    for q in names:
        # The mask must matter, or the comparison proves nothing.
        assert (causal[q] - expected[q]).abs().max() > 5 * ATOL, q

    def params(prefix_len: int | None) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprob_token_ids=list(range(NUM_SCORED)),
            extra_args=None if prefix_len is None else {"prefix_lm_len": prefix_len},
        )

    # Room for every prompt in one step, so the reads share a batch.
    with vllm_runner(
        checkpoint,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as runner:
        llm = runner.llm
        vllm_config = llm.llm_engine.vllm_config
        assert vllm_config.model_config.architecture == "Qwen3_5ForSlotDecision"
        assert vllm_config.model_config.head_dtype == torch.float32
        # Prefix caching stays on with FlashInfer's GDN prefill only.
        prefix_caching = vllm_config.cache_config.enable_prefix_caching
        if prefix_caching:
            scheduler_cls = vllm_config.scheduler_config.scheduler_cls
            assert scheduler_cls.startswith("vllm.v1.core.sched.prefix_lm_scheduler.")

        # The three questions with their blocks, then q0 without a prefix length,
        # which is read causally.
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=questions[q]) for q in [*names, "q0"]],
            [params(p)] * len(names) + [params(None)],
            use_tqdm=False,
        )
        references = [*(expected[q] for q in names), causal["q0"]]
        others = [*(causal[q] for q in names), expected["q0"]]

        def check(name: str, out, reference, other) -> None:
            logprobs = out.outputs[0].logprobs[0]
            got = torch.tensor([logprobs[t].logprob for t in range(NUM_SCORED)])
            error = (got - reference).abs().max().item()
            gap = (other - reference).abs().max().item()
            assert error < ATOL, (name, error)
            assert error < MAX_ERROR_VS_MASK_GAP * gap, (name, error, gap)

        for args in zip([*names, "q0 causal"], outputs, references, others):
            check(*args)

        if not prefix_caching:
            return
        # The prefix alone, as the strategy's warm-up sends it. The questions
        # then reuse its cache, and never past it: q1 and q2 share 40 more
        # tokens, and q2 comes twice.
        llm.generate(TokensPrompt(prompt_token_ids=prefix), params(p), use_tqdm=False)
        for name in ("q1", "q2", "q2"):
            (out,) = llm.generate(
                TokensPrompt(prompt_token_ids=questions[name]),
                params(p),
                use_tqdm=False,
            )
            check(f"{name} cached", out, expected[name], causal[name])
            assert 0 < (out.num_cached_tokens or 0) <= p, out.num_cached_tokens
