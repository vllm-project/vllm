# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile

import mteb
import numpy as np
from mteb.models import ModelMeta
from torch.utils.data import DataLoader

MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["eng"]


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


def prompt_template(q, p):
    return f"question:{q} \n \n passage:{p}"


def patch_cross_encoder(hf_model):
    org_tokenizer = hf_model.tokenizer

    class Tokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, pairs, **kwargs):
            text = [prompt_template(*pair) for pair in pairs]
            return self.tokenizer(text, **kwargs)

    hf_model.tokenizer = Tokenizer(org_tokenizer)


class ScoreClientMtebEncoder(mteb.CrossEncoderProtocol):
    mteb_model_meta = _empty_model_meta

    def __init__(self, hf_model):
        self.hf_model = hf_model
        patch_cross_encoder(hf_model)

    def predict(
        self,
        inputs1: DataLoader[mteb.types.BatchedInput],
        inputs2: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        queries = [text for batch in inputs1 for text in batch["text"]]
        full_corpus = [text for batch in inputs2 for text in batch["text"]]
        pairs = list(zip(queries, full_corpus))
        return self.hf_model.predict(pairs, batch_size=1)


def run_mteb_rerank(cross_encoder: mteb.CrossEncoderProtocol, tasks, languages):
    with tempfile.TemporaryDirectory() as prediction_folder:
        bm25s = mteb.get_model("bm25s")
        eval_splits = ["test"]

        mteb_tasks: list[mteb.abstasks.AbsTaskRetrieval] = mteb.get_tasks(
            tasks=tasks, languages=languages, eval_splits=eval_splits
        )

        mteb.evaluate(
            bm25s,
            mteb_tasks,
            prediction_folder=prediction_folder,
            show_progress_bar=False,
            # don't save results for test runs
            cache=None,
            overwrite_strategy="always",
        )

        second_stage_tasks = []
        for task in mteb_tasks:
            second_stage_tasks.append(
                task.convert_to_reranking(
                    prediction_folder,
                    top_k=10,
                )
            )

        results = mteb.evaluate(
            cross_encoder,
            second_stage_tasks,
            show_progress_bar=False,
            cache=None,
        )
        main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def get_st_main_score(model_name):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        model_name,
        trust_remote_code=True,
    )

    st_main_score = run_mteb_rerank(
        ScoreClientMtebEncoder(model),
        tasks=MTEB_RERANK_TASKS,
        languages=MTEB_RERANK_LANGS,
    )
    st_dtype = next(model.model.model.parameters()).dtype
    return st_main_score, st_dtype


def run(model_name):
    st_main_score, model_dtype = get_st_main_score(model_name)
    print(model_name, model_dtype, st_main_score)


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1]
    run(model_name)
