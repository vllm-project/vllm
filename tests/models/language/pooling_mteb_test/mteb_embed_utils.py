# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mteb
import numpy as np
import torch
from mteb.models import ModelMeta
from mteb.types import Array
from torch.utils.data import DataLoader

import tests.ci_envs as ci_envs
from tests.models.utils import (
    EmbedModelInfo,
    check_embeddings_close,
    get_vllm_extra_kwargs,
)

# Most embedding models on the STS12 task (See #17175):
# - Model implementation and minor changes in tensor dtype
#   results in differences less than 1e-4
# - Different model results in differences more than 1e-3
# 1e-4 is a good tolerance threshold
MTEB_EMBED_TASKS = ["STS12"]
MTEB_EMBED_TOL = 1e-4


_empty_model_meta = ModelMeta(
    loader=None,
    name="vllm/model",
    revision="1",
    release_date=None,
    languages=None,
    framework=[],
    similarity_fn_name=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=None,
    training_datasets=None,
    modalities=["text"],  # 'image' can be added to evaluate multimodal models
)


class MtebEmbedMixin(mteb.EncoderProtocol):
    mteb_model_meta = _empty_model_meta

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        sim = np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)
        return sim

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        sim = np.sum(embeddings1 * embeddings2, axis=1) / (
            norm1.flatten() * norm2.flatten()
        )
        return sim


class VllmMtebEncoder(MtebEmbedMixin):
    def __init__(self, vllm_model):
        self.llm = vllm_model
        self.rng = np.random.default_rng(seed=42)

    def encode(
        self,
        inputs: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Hoping to discover potential scheduling
        # issues by randomizing the order.
        sentences = [text for batch in inputs for text in batch["text"]]
        r = self.rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]
        outputs = self.llm.embed(sentences, use_tqdm=False)
        embeds = np.array(outputs)
        embeds = embeds[np.argsort(r)]
        return embeds


class OpenAIClientMtebEncoder(MtebEmbedMixin):
    def __init__(self, model_name: str, client):
        self.model_name = model_name
        self.client = client
        self.rng = np.random.default_rng(seed=42)

    def encode(
        self,
        inputs: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Hoping to discover potential scheduling
        # issues by randomizing the order.
        sentences = [text for batch in inputs for text in batch["text"]]
        r = self.rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]

        embeddings = self.client.embeddings.create(
            model=self.model_name, input=sentences
        )
        outputs = [d.embedding for d in embeddings.data]
        embeds = np.array(outputs)
        embeds = embeds[np.argsort(r)]
        return embeds


def run_mteb_embed_task(encoder: mteb.EncoderProtocol, tasks):
    tasks = mteb.get_tasks(tasks=tasks)
    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )

    main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def mteb_test_embed_models(
    hf_runner,
    vllm_runner,
    model_info: EmbedModelInfo,
    vllm_extra_kwargs=None,
    hf_model_callback=None,
    atol=MTEB_EMBED_TOL,
):
    vllm_extra_kwargs = get_vllm_extra_kwargs(model_info, vllm_extra_kwargs)

    # Test embed_dims, isnan and whether to use normalize
    example_prompts = ["The chef prepared a delicious meal." * 1000]

    with vllm_runner(
        model_info.name,
        runner="pooling",
        max_model_len=model_info.max_model_len,
        **vllm_extra_kwargs,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config

        # Confirm whether vllm is using the correct architecture
        if model_info.architecture:
            assert model_info.architecture in model_config.architectures

        # Confirm whether the important configs in model_config are correct.
        if model_info.pooling_type is not None:
            assert model_config.pooler_config.pooling_type == model_info.pooling_type
        if model_info.attn_type is not None:
            assert model_config.attn_type == model_info.attn_type
        if model_info.is_prefix_caching_supported is not None:
            assert (
                model_config.is_prefix_caching_supported
                == model_info.is_prefix_caching_supported
            )
        if model_info.is_chunked_prefill_supported is not None:
            assert (
                model_config.is_chunked_prefill_supported
                == model_info.is_chunked_prefill_supported
            )

        vllm_main_score = run_mteb_embed_task(
            VllmMtebEncoder(vllm_model), MTEB_EMBED_TASKS
        )
        vllm_dtype = vllm_model.llm.llm_engine.model_config.dtype
        head_dtype = model_config.head_dtype

        # Test embedding_size, isnan and whether to use normalize
        vllm_outputs = vllm_model.embed(example_prompts, truncate_prompt_tokens=-1)
        outputs_tensor = torch.tensor(vllm_outputs)
        assert not torch.any(torch.isnan(outputs_tensor))
        embedding_size = model_config.embedding_size
        assert torch.tensor(vllm_outputs).shape[-1] == embedding_size

    # Accelerate mteb test by setting
    # SentenceTransformers mteb score to a constant
    if model_info.mteb_score is None:
        with hf_runner(
            model_info.name,
            is_sentence_transformer=True,
            dtype=ci_envs.VLLM_CI_HF_DTYPE or model_info.hf_dtype,
        ) as hf_model:
            # e.g. setting default parameters for the encode method of hf_runner
            if hf_model_callback is not None:
                hf_model_callback(hf_model)

            st_main_score = run_mteb_embed_task(hf_model, MTEB_EMBED_TASKS)
            st_dtype = next(hf_model.model.parameters()).dtype

            # Check embeddings close to hf outputs
            hf_outputs = hf_model.encode(example_prompts)
            check_embeddings_close(
                embeddings_0_lst=hf_outputs,
                embeddings_1_lst=vllm_outputs,
                name_0="hf",
                name_1="vllm",
                tol=1e-2,
            )
    else:
        st_main_score = model_info.mteb_score
        st_dtype = "Constant"

    print("Model:", model_info.name)
    print("VLLM:", f"dtype:{vllm_dtype}", f"head_dtype:{head_dtype}", vllm_main_score)
    print("SentenceTransformers:", st_dtype, st_main_score)
    print("Difference:", st_main_score - vllm_main_score)

    # We are not concerned that the vllm mteb results are better
    # than SentenceTransformers, so we only perform one-sided testing.
    assert st_main_score - vllm_main_score < atol
