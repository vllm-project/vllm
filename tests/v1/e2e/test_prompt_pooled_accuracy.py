# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for the runner-native return_embed flag.

Compare the runner-native mean-pool (opted-in via
``SamplingParams.extra_args["return_embed"] = True``) against vLLM's own
``embed`` task with ``pooling_type="MEAN"``, which mean-pools the same
hidden states through the standard pooler path. Both paths
should agree within bf16 precision.

Also exercises LoRA: the runner-native pool reads hidden states
that already include LoRA deltas, so it should match the embed task with
the same adapter applied.
"""

import gc

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_MODEL = "Qwen/Qwen3-0.6B"
LLAMA_LORA_PATH = "jeeejeee/llama32-3b-text2sql-spider"


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "accelerator") and torch.accelerator.is_available()),
    reason="Accelerator (CUDA) required",
)


def _opt_in(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"return_embed": True},
    )


def _embedding(out) -> list[float] | None:
    if out.kv_transfer_params is None:
        return None
    return out.kv_transfer_params.get("embed")


def _get_vllm_mean_pooled_embeddings(
    model_name: str,
    prompts: list[str],
    lora_path: str | None = None,
    max_lora_rank: int = 8,
) -> list[torch.Tensor]:
    """Run vLLM's embed task with MEAN pooling as a reference.

    The runner-native pool and the embed task both operate on the same
    hidden states from ``LlamaForCausalLM`` / ``Qwen3ForCausalLM``,
    so their mean-pooled outputs should match (modulo dtype precision).
    """
    llm_kwargs: dict = {}
    lora_request = None
    if lora_path is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(
        model=model_name,
        runner="pooling",
        convert="embed",
        pooler_config={
            "pooling_type": "MEAN",
            "use_activation": False,
        },
        max_model_len=128,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        **llm_kwargs,
    )
    if lora_path is not None:
        lora_request = LoRARequest("ref_lora", 1, lora_path)
        llm.llm_engine.add_lora(lora_request)

    outputs = llm.embed(prompts, lora_request=lora_request, use_tqdm=False)
    embeddings = [torch.tensor(o.outputs.embedding) for o in outputs]
    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    return embeddings


def _get_runner_native_embeddings(
    model_name: str,
    prompts: list[str],
    lora_path: str | None = None,
    max_lora_rank: int = 8,
    max_tokens: int = 8,
) -> list[list[float]]:
    """Run the generate task with ``return_embed`` opted in."""
    llm_kwargs: dict = {}
    lora_request = None
    if lora_path is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(
        model=model_name,
        max_model_len=128,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        enable_return_embed=True,
        **llm_kwargs,
    )
    if lora_path is not None:
        lora_request = LoRARequest("text2sql", 1, lora_path)
        llm.llm_engine.add_lora(lora_request)

    outputs = llm.generate(
        prompts,
        _opt_in(max_tokens=max_tokens),
        lora_request=lora_request,
        use_tqdm=False,
    )
    pooled = []
    for o in outputs:
        emb = _embedding(o)
        assert emb is not None, "expected 'embed' in kv_transfer_params"
        pooled.append(emb)
    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    return pooled


@pytest.mark.parametrize(
    "model_name",
    [
        LLAMA_MODEL,
        QWEN_MODEL,
    ],
)
@torch.inference_mode()
def test_runner_pool_accuracy_vs_embed_task(model_name: str):
    """Compare runner-native mean-pool against vLLM's embed-task MEAN pooler.

    The runner accumulates hidden states for prompt tokens and
    divides by ``prompt_len``. The embed task applies ``MeanPool`` to the
    same hidden states. Outputs should match within bf16 precision.
    """
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # Reference: vLLM embed task with MEAN pooling.
    embed_results = _get_vllm_mean_pooled_embeddings(model_name, prompts)

    # System under test: generate task with return_embed opted in.
    runner_results = _get_runner_native_embeddings(model_name, prompts)

    assert len(runner_results) == len(prompts)

    for i, (pooled_list, ref) in enumerate(zip(runner_results, embed_results)):
        runner_pooled = torch.tensor(pooled_list)
        assert runner_pooled.shape == ref.shape, (
            f"Prompt {i}: shape mismatch: "
            f"runner {runner_pooled.shape} vs embed {ref.shape}"
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            runner_pooled.float().unsqueeze(0),
            ref.float().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, (
            f"Prompt {i} ({model_name}): cosine similarity {cos_sim:.6f} "
            f"below threshold 0.999."
        )


@torch.inference_mode()
def test_runner_pool_with_lora_vs_embed_task():
    """Compare LoRA-applied runner-native pool against LoRA-applied embed.

    Same comparison as ``test_runner_pool_accuracy_vs_embed_task`` but with
    the ``jeeejeee/llama32-3b-text2sql-spider`` adapter applied to every
    request on both sides. Confirms the runner-native pool reflects
    LoRA-modified hidden states and matches the standard MeanPool pooler
    when LoRA is active.
    """
    hidden_size = 3072  # Llama-3.2-3B-Instruct
    prompts = [
        "Translate to SQL: list the names of all employees.",
        "Translate to SQL: count the number of orders per customer.",
    ]

    # Reference: vLLM embed task with MEAN pooling, LoRA applied.
    embed_results = _get_vllm_mean_pooled_embeddings(
        LLAMA_MODEL, prompts, lora_path=LLAMA_LORA_PATH, max_lora_rank=8
    )

    # System under test: generate task with return_embed, LoRA applied.
    runner_results = _get_runner_native_embeddings(
        LLAMA_MODEL,
        prompts,
        lora_path=LLAMA_LORA_PATH,
        max_lora_rank=8,
        max_tokens=32,
    )

    assert len(runner_results) == len(prompts)

    for i, (pooled_list, ref) in enumerate(zip(runner_results, embed_results)):
        runner_pooled = torch.tensor(pooled_list)
        assert runner_pooled.shape == (hidden_size,), (
            f"Prompt {i}: shape {runner_pooled.shape} != ({hidden_size},)"
        )
        assert runner_pooled.shape == ref.shape

        cos_sim = torch.nn.functional.cosine_similarity(
            runner_pooled.float().unsqueeze(0),
            ref.float().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, (
            f"Prompt {i}: LoRA cosine similarity {cos_sim:.6f} below 0.999"
        )
