# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from functools import partial

import mteb
import numpy as np
import torch
from mteb.models import ModelMeta
from mteb.types import Array
from torch.utils.data import DataLoader

MTEB_EMBED_TASKS = ["STS12"]

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


class MtebEncoderMixin(mteb.EncoderProtocol):
    def encode(
        self,
        inputs: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

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


class HFMtebEncoder(MtebEncoderMixin):
    mteb_model_meta = _empty_model_meta

    def __init__(self, hf_model):
        self.model = hf_model

    def encode(
        self,
        inputs: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        sentences = [text for batch in inputs for text in batch["text"]]
        embeds = self.model.encode(sentences)
        return embeds


def run_mteb_embed_task(encoder: mteb.EncoderProtocol):
    tasks = mteb.get_tasks(tasks=MTEB_EMBED_TASKS)
    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )

    main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def get_st_main_score(model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)
    model_dtype = next(model.parameters()).dtype

    model_encode = model.encode

    def encode(sentences, **kwargs):
        kwargs.pop("prompt_name", None)
        return model_encode(sentences, **kwargs)

    model.encode = encode

    if model_name == "jinaai/jina-embeddings-v3":
        model.encode = partial(model.encode, task="text-matching")

    main_score = run_mteb_embed_task(HFMtebEncoder(model))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return main_score, model_dtype


def run(model_name):
    st_main_score, model_dtype = get_st_main_score(model_name)
    print(model_name, model_dtype, st_main_score)


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1]
    run(model_name)
