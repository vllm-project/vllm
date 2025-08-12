# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import math
from array import array

import openai
import pytest
from scipy.spatial.distance import cosine

from vllm import LLM, SamplingParams
from vllm.config import ModelConfig
from vllm.utils import STR_BACKEND_ENV_VAR

from ....utils import RemoteOpenAIServer

# GritLM embedding implementation is only supported by XFormers backend.
pytestmark = pytest.mark.skipif(not importlib.util.find_spec("xformers"),
                                reason="GritLM requires XFormers")

MODEL_NAME = "parasail-ai/GritLM-7B-vllm"
MAX_MODEL_LEN = 4000


def _arr(arr):
    """
    Convert a list of integers to an array of integers.
    """
    return array("i", arr)


def test_find_array():
    from vllm.model_executor.models.gritlm import GritLMPooler

    model_config = ModelConfig(
        MODEL_NAME,
        task="embed",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
    )
    pooler = GritLMPooler(model_config=model_config)

    arr = _arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert pooler._find_array(arr, _arr([3, 4, 5]), start_idx=0) == 3
    assert pooler._find_array(arr, _arr([3, 4, 5]), start_idx=1) == 3
    assert pooler._find_array(arr, _arr([3, 4, 5]), start_idx=5) == -1
    assert pooler._find_array(arr, _arr([3, 5]), start_idx=0) == -1

    with pytest.raises(ValueError):
        pooler._find_array(arr, _arr([3, 4, 5]), start_idx=-1)


def run_llm_encode(
    llm: LLM,
    queries: list[str],
    instruction: str,
) -> list[list[float]]:
    outputs = llm.embed([instruction + q for q in queries])
    return [output.outputs.embedding for output in outputs]


async def run_client_embeddings(
    client: openai.AsyncOpenAI,
    queries: list[str],
    instruction: str,
) -> list[list[float]]:
    outputs = await client.embeddings.create(
        model=MODEL_NAME,
        input=[instruction + q for q in queries],
    )
    return [data.embedding for data in outputs.data]


def gritlm_instruction(instruction):
    return ("<|user|>\n" + instruction +
            "\n<|embed|>\n" if instruction else "<|embed|>\n")


def get_test_data():
    """
    Grabbed this test data and the expected values from
    README.md in https://github.com/ContextualAI/gritlm
    """
    q_instruction = gritlm_instruction(
        "Given a scientific paper title, retrieve the paper's abstract", )
    queries = [
        "Bitcoin: A Peer-to-Peer Electronic Cash System",
        "Generative Representational Instruction Tuning",
    ]

    d_instruction = gritlm_instruction("")
    documents = [
        # ruff: noqa: E501
        "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power. As long as a majority of CPU power is controlled by nodes that are not cooperating to attack the network, they'll generate the longest chain and outpace attackers. The network itself requires minimal structure. Messages are broadcast on a best effort basis, and nodes can leave and rejoin the network at will, accepting the longest proof-of-work chain as proof of what happened while they were gone.",
        "All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8X7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm.",
    ]

    return queries, q_instruction, documents, d_instruction


def validate_embed_output(q_rep: list[list[float]], d_rep: list[list[float]]):
    cosine_sim_q0_d0 = 1 - cosine(q_rep[0], d_rep[0])
    assert math.isclose(cosine_sim_q0_d0, 0.609, abs_tol=0.001)

    cosine_sim_q0_d1 = 1 - cosine(q_rep[0], d_rep[1])
    assert math.isclose(cosine_sim_q0_d1, 0.101, abs_tol=0.001)

    cosine_sim_q1_d0 = 1 - cosine(q_rep[1], d_rep[0])
    assert math.isclose(cosine_sim_q1_d0, 0.120, abs_tol=0.001)

    cosine_sim_q1_d1 = 1 - cosine(q_rep[1], d_rep[1])
    assert math.isclose(cosine_sim_q1_d1, 0.534, abs_tol=0.001)


def test_gritlm_offline_embedding(monkeypatch: pytest.MonkeyPatch,
                                  vllm_runner):
    # GritLM embedding implementation is only supported by XFormers backend.
    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, "XFORMERS")

        queries, q_instruction, documents, d_instruction = get_test_data()

        with vllm_runner(
                MODEL_NAME,
                task="embed",
                max_model_len=MAX_MODEL_LEN,
        ) as vllm_model:
            llm = vllm_model.model

            d_rep = run_llm_encode(
                llm,
                documents,
                d_instruction,
            )
            q_rep = run_llm_encode(
                llm,
                queries,
                q_instruction,
            )

        validate_embed_output(q_rep, d_rep)


@pytest.mark.asyncio
async def test_gritlm_api_server_embedding():
    queries, q_instruction, documents, d_instruction = get_test_data()

    # GritLM embedding implementation is only supported by XFormers backend.
    args = ["--task", "embed", "--max_model_len", str(MAX_MODEL_LEN)]
    env_dict = {STR_BACKEND_ENV_VAR: "XFORMERS"}

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as server:
        client_embedding = server.get_async_client()

        d_rep = await run_client_embeddings(
            client_embedding,
            documents,
            d_instruction,
        )
        q_rep = await run_client_embeddings(
            client_embedding,
            queries,
            q_instruction,
        )

    validate_embed_output(q_rep, d_rep)


def test_gritlm_offline_generate(monkeypatch: pytest.MonkeyPatch, vllm_runner):
    # GritLM embedding implementation is only supported by XFormers backend.
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
        m.setenv(STR_BACKEND_ENV_VAR, "XFORMERS")

        input = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"

        with vllm_runner(
                MODEL_NAME,
                task="generate",
                max_model_len=MAX_MODEL_LEN,
        ) as vllm_model:
            llm = vllm_model.model

            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
            outputs = llm.generate(input, sampling_params=sampling_params)

        assert outputs[0].outputs[0].text == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_gritlm_api_server_generate():
    input = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"

    # GritLM embedding implementation is only supported by XFormers backend.
    args = ["--task", "generate", "--max_model_len", str(MAX_MODEL_LEN)]
    env_dict = {"VLLM_USE_V1": "0", STR_BACKEND_ENV_VAR: "XFORMERS"}

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as server:
        client_generate = server.get_async_client()

        outputs = await client_generate.completions.create(
            model=MODEL_NAME,
            prompt=input,
            max_tokens=256,
            temperature=0.0,
        )

    assert outputs.choices[0].text == "The capital of France is Paris."
