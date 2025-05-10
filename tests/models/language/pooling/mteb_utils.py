# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import mteb
import numpy as np

MTEB_EMBED_TASKS = ["STS12"]


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
