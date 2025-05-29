# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import mteb
import numpy as np
import pytest

from tests.models.utils import EmbedModelInfo
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

# Most models on the STS12 task (See #17175):
# - Model implementation and minor changes in tensor dtype
#   results in differences less than 1e-4
# - Different model results in differences more than 1e-3
# 1e-4 is a good tolerance threshold
MTEB_EMBED_TASKS = ["STS12"]
MTEB_EMBED_TOL = 1e-4


class VllmMtebEncoder(mteb.Encoder):

    def __init__(self, vllm_model):
        super().__init__()
        self.model = vllm_model
        self.rng = np.random.default_rng(seed=42)

    def encode(
        self,
        sentences: Sequence[str],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Hoping to discover potential scheduling
        # issues by randomizing the order.
        r = self.rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]
        outputs = self.model.encode(sentences, use_tqdm=False)
        embeds = np.array(outputs)
        embeds = embeds[np.argsort(r)]
        return embeds


class OpenAIClientMtebEncoder(mteb.Encoder):

    def __init__(self, model_name: str, client):
        super().__init__()
        self.model_name = model_name
        self.client = client
        self.rng = np.random.default_rng(seed=42)

    def encode(self, sentences: Sequence[str], *args, **kwargs) -> np.ndarray:
        # Hoping to discover potential scheduling
        # issues by randomizing the order.
        r = self.rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]

        embeddings = self.client.embeddings.create(model=self.model_name,
                                                   input=sentences)
        outputs = [d.embedding for d in embeddings.data]
        embeds = np.array(outputs)
        embeds = embeds[np.argsort(r)]
        return embeds


def run_mteb_embed_task(encoder, tasks):
    tasks = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(encoder, verbosity=0, output_folder=None)

    main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def run_mteb_embed_task_st(model_name, tasks):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return run_mteb_embed_task(model, tasks)


def mteb_test_embed_models(hf_runner,
                           vllm_runner,
                           model_info: EmbedModelInfo,
                           vllm_extra_kwargs=None,
                           hf_model_callback=None):
    if not model_info.enable_test:
        # A model family has many models with the same architecture,
        # and we don't need to test each one.
        pytest.skip("Skipping test.")

    vllm_extra_kwargs = vllm_extra_kwargs or {}
    vllm_extra_kwargs["dtype"] = model_info.dtype

    with vllm_runner(model_info.name,
                     task="embed",
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:

        if model_info.architecture:
            assert (model_info.architecture
                    in vllm_model.model.llm_engine.model_config.architectures)

        vllm_main_score = run_mteb_embed_task(VllmMtebEncoder(vllm_model),
                                              MTEB_EMBED_TASKS)
        vllm_dtype = vllm_model.model.llm_engine.model_config.dtype
        model_dtype = getattr(
            vllm_model.model.llm_engine.model_config.hf_config, "torch_dtype",
            vllm_dtype)

    with set_default_torch_dtype(model_dtype) and hf_runner(
            model_info.name, is_sentence_transformer=True,
            dtype=model_dtype) as hf_model:

        if hf_model_callback is not None:
            hf_model_callback(hf_model)

        st_main_score = run_mteb_embed_task(hf_model, MTEB_EMBED_TASKS)

    print("VLLM:", vllm_dtype, vllm_main_score)
    print("SentenceTransformer:", model_dtype, st_main_score)
    print("Difference:", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_EMBED_TOL)
