import math
import os
from typing import List

import openai
import pytest
import pytest_asyncio
from scipy.spatial.distance import cosine

import vllm

from ....utils import RemoteOpenAIServer

MODEL_NAME = "parasail-ai/GritLM-7B-vllm"

# GritLM implementation is only supported by XFormers backend.
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "embedding",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def run_llm_encode(llm: vllm.LLM, queries: List[str], instruction: str,
                   use_instruction_arg: bool) -> List[float]:
    pooling_params = vllm.PoolingParams(
        additional_data={"instruction_seq": instruction
                         }) if use_instruction_arg else None
    outputs = llm.encode(
        [instruction + q for q in queries],
        pooling_params=pooling_params,
    )
    return [output.outputs.embedding for output in outputs]


async def run_client_embeddings(client: vllm.LLM, queries: List[str],
                                instruction: str,
                                use_instruction_arg: bool) -> List[float]:
    additional_data = {
        "instruction_seq": instruction
    } if use_instruction_arg else None
    outputs = await client.embeddings.create(
        model=MODEL_NAME,
        input=[instruction + q for q in queries],
        extra_body={"additional_data": additional_data},
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
        "Given a scientific paper title, retrieve the paper's abstract")
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


def validate_output(q_rep: List[float], d_rep: List[float]):
    cosine_sim_q0_d0 = 1 - cosine(q_rep[0], d_rep[0])
    assert math.isclose(cosine_sim_q0_d0, 0.609, abs_tol=0.001)

    cosine_sim_q0_d1 = 1 - cosine(q_rep[0], d_rep[1])
    assert math.isclose(cosine_sim_q0_d1, 0.101, abs_tol=0.001)

    cosine_sim_q1_d0 = 1 - cosine(q_rep[1], d_rep[0])
    assert math.isclose(cosine_sim_q1_d0, 0.120, abs_tol=0.001)

    cosine_sim_q1_d1 = 1 - cosine(q_rep[1], d_rep[1])
    assert math.isclose(cosine_sim_q1_d1, 0.532, abs_tol=0.001)


@pytest.mark.parametrize("use_instruction_arg", [True, False])
def test_gritlm_offline(use_instruction_arg: bool):
    queries, q_instruction, documents, d_instruction = get_test_data()

    llm = vllm.LLM(MODEL_NAME, task="embedding")

    d_rep = run_llm_encode(
        llm,
        documents,
        d_instruction,
        use_instruction_arg=use_instruction_arg,
    )
    q_rep = run_llm_encode(
        llm,
        queries,
        q_instruction,
        use_instruction_arg=use_instruction_arg,
    )

    validate_output(q_rep, d_rep)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_instruction_arg", [True, False])
async def test_gritlm_api_server(client: openai.AsyncOpenAI,
                                 use_instruction_arg: bool):
    queries, q_instruction, documents, d_instruction = get_test_data()

    d_rep = await run_client_embeddings(
        client,
        documents,
        d_instruction,
        use_instruction_arg=use_instruction_arg,
    )
    q_rep = await run_client_embeddings(
        client,
        queries,
        q_instruction,
        use_instruction_arg=use_instruction_arg,
    )

    validate_output(q_rep, d_rep)
