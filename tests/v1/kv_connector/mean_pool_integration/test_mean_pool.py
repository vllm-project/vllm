# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MeanPoolHiddenStatesConnector.

1. End-to-end accuracy tests for Llama-3.2-3B-Instruct and Qwen3-0.6B that
   compare the connector's mean-pooled hidden states against vLLM's own
   embed task (``runner="pooling"`` + ``pooling_type="MEAN"``), which
   mean-pools the same hidden states using the standard pooler
   path.
2. LoRA test for Llama-3.2-3B-Instruct using the
   ``jeeejeee/llama32-3b-text2sql-spider`` adapter, verifying the
   connector keeps working when LoRA is applied at request time.
"""

import gc
import os
import tempfile

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Defaults are HF repo IDs and resolve from ~/.cache/huggingface/hub.
LLAMA_MODEL = os.environ.get(
    "VLLM_TEST_LLAMA_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
)
QWEN_MODEL = os.environ.get("VLLM_TEST_QWEN_MODEL", "Qwen/Qwen3-0.6B")
LLAMA_LORA_PATH = os.environ.get(
    "VLLM_TEST_LLAMA_LORA_PATH", "jeeejeee/llama32-3b-text2sql-spider"
)


# =====================================================================
# End-to-end accuracy: connector vs. vLLM's own embed (MEAN) task
# =====================================================================


def _get_vllm_mean_pooled_embeddings(
    model_name: str,
    prompts: list[str],
    lora_path: str | None = None,
    max_lora_rank: int = 8,
) -> list[torch.Tensor]:
    """Run vLLM's embed task with MEAN pooling as a reference.

    Both the connector and the embed task operate on the
    hidden states from ``LlamaForCausalLM`` / ``Qwen3ForCausalLM``, so
    their mean-pooled outputs should match (modulo dtype precision).

    If ``lora_path`` is provided, the LoRA adapter is loaded and applied to
    every embed request, so the reference reflects LoRA-modified hidden
    states.
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


@pytest.mark.parametrize(
    "model_name",
    [
        LLAMA_MODEL,
        QWEN_MODEL,
    ],
)
@torch.inference_mode()
def test_mean_pool_accuracy_vs_embed_task(model_name: str):
    """Compare connector mean-pool against vLLM's embed-task MEAN pooler.

    The connector caches hidden states (via the embedded
    ``CacheOnlyAttentionLayer``) and mean-pools over prompt tokens. The
    embed task applies ``MeanPool`` to the same hidden states.
    Outputs should match within bf16 precision.
    """
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # Reference: vLLM embed task with MEAN pooling.
    embed_results = _get_vllm_mean_pooled_embeddings(model_name, prompts)

    # System under test: generate task with MeanPoolHiddenStatesConnector.
    with tempfile.TemporaryDirectory() as storage_path:
        llm = LLM(
            model=model_name,
            kv_transfer_config={
                "kv_connector": "MeanPoolHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": storage_path,
                },
            },
            max_model_len=128,
            enforce_eager=True,
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
        outputs = llm.generate(prompts, sampling_params)
        del llm
        gc.collect()
        torch.accelerator.empty_cache()

    assert len(outputs) == len(prompts)

    for i, (output, ref) in enumerate(zip(outputs, embed_results)):
        assert output.kv_transfer_params is not None, (
            f"Prompt {i}: kv_transfer_params should not be None"
        )
        pooled_list = output.kv_transfer_params.get("mean_pooled_hidden_states")
        assert pooled_list is not None, (
            f"Prompt {i}: mean_pooled_hidden_states not found"
        )

        connector_pooled = torch.tensor(pooled_list)
        assert connector_pooled.shape == ref.shape, (
            f"Prompt {i}: shape mismatch: "
            f"connector {connector_pooled.shape} vs embed {ref.shape}"
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            connector_pooled.float().unsqueeze(0),
            ref.float().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, (
            f"Prompt {i} ({model_name}): cosine similarity {cos_sim:.6f} "
            f"below threshold 0.999."
        )


# =====================================================================
# LoRA: connector still returns pooled hidden states with adapters
# =====================================================================


@torch.inference_mode()
def test_mean_pool_with_lora_vs_embed_task():
    """Compare LoRA-applied connector mean-pool against LoRA-applied embed.

    Same comparison as ``test_mean_pool_accuracy_vs_embed_task`` but with the
    ``jeeejeee/llama32-3b-text2sql-spider`` adapter applied to every request
    on both sides. Confirms the connector reflects LoRA-modified
    hidden states and matches what the standard MeanPool pooler produces
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

    # System under test: generate task with the connector, LoRA applied.
    with tempfile.TemporaryDirectory() as storage_path:
        llm = LLM(
            model=LLAMA_MODEL,
            kv_transfer_config={
                "kv_connector": "MeanPoolHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": storage_path,
                },
            },
            enable_lora=True,
            max_loras=1,
            max_lora_rank=8,
            max_model_len=128,
            enforce_eager=True,
            dtype="bfloat16",
        )

        lora_request = LoRARequest("text2sql", 1, LLAMA_LORA_PATH)
        llm.llm_engine.add_lora(lora_request)
        assert 1 in llm.llm_engine.list_loras()

        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        del llm
        gc.collect()
        torch.accelerator.empty_cache()

    assert len(outputs) == len(prompts)

    for i, (output, ref) in enumerate(zip(outputs, embed_results)):
        assert output.kv_transfer_params is not None, (
            f"Prompt {i}: kv_transfer_params should not be None"
        )
        pooled_list = output.kv_transfer_params.get("mean_pooled_hidden_states")
        assert pooled_list is not None, (
            f"Prompt {i}: mean_pooled_hidden_states not found"
        )

        connector_pooled = torch.tensor(pooled_list)
        assert connector_pooled.shape == (hidden_size,), (
            f"Prompt {i}: shape {connector_pooled.shape} != ({hidden_size},)"
        )
        assert connector_pooled.shape == ref.shape

        cos_sim = torch.nn.functional.cosine_similarity(
            connector_pooled.float().unsqueeze(0),
            ref.float().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, (
            f"Prompt {i}: LoRA cosine similarity {cos_sim:.6f} below 0.999"
        )
