import pytest

from ..utils import RemoteOpenAIServer


@pytest.mark.parametrize(
    "TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL, MODEL_NAME", [
        (2, 2, 0, 1, "meta-llama/Meta-Llama-3-8B"),
        (2, 2, 1, 0, "meta-llama/Meta-Llama-3-8B"),
        (1, 3, 0, 0, "meta-llama/Meta-Llama-3-8B"),
        (1, 4, 0, 1, "meta-llama/Meta-Llama-3-8B"),
        (1, 4, 1, 0, "meta-llama/Meta-Llama-3-8B"),
    ])
def test_compare_tp(TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL, MODEL_NAME):
    pp_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--distributed-executor-backend",
        "ray",
    ]

    # compare without pipeline parallelism
    # NOTE: use mp backend for TP
    # PP tests might involve multiple nodes, and ray might
    #  schedule all workers in a node other than the head node,
    #  which can cause the test to fail.
    tp_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(max(TP_SIZE, 2)),  # use at least TP_SIZE=2 to hold the model
        "--distributed-executor-backend",
        "mp",
    ]
    if CHUNKED_PREFILL:
        pp_args.append("--enable-chunked-prefill")
        tp_args.append("--enable-chunked-prefill")
    if EAGER_MODE:
        pp_args.append("--enforce-eager")
        tp_args.append("--enforce-eager")

    results = []
    for args in [pp_args, tp_args]:
        with RemoteOpenAIServer(MODEL_NAME, args) as server:
            client = server.get_client()

            # test models list
            models = client.models.list()
            models = models.data
            served_model = models[0]
            results.append({
                "test": "models_list",
                "id": served_model.id,
                "root": served_model.root,
            })

            # test with text prompt
            completion = client.completions.create(model=MODEL_NAME,
                                                   prompt="Hello, my name is",
                                                   max_tokens=5,
                                                   temperature=0.0)

            results.append({
                "test": "single_completion",
                "text": completion.choices[0].text,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": completion.usage,
            })

            # test using token IDs
            completion = client.completions.create(
                model=MODEL_NAME,
                prompt=[0, 0, 0, 0, 0],
                max_tokens=5,
                temperature=0.0,
            )

            results.append({
                "test": "token_ids",
                "text": completion.choices[0].text,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": completion.usage,
            })

            # test simple list
            batch = client.completions.create(
                model=MODEL_NAME,
                prompt=["Hello, my name is", "Hello, my name is"],
                max_tokens=5,
                temperature=0.0,
            )

            results.append({
                "test": "simple_list",
                "text0": batch.choices[0].text,
                "text1": batch.choices[1].text,
            })

            # test streaming
            batch = client.completions.create(
                model=MODEL_NAME,
                prompt=["Hello, my name is", "Hello, my name is"],
                max_tokens=5,
                temperature=0.0,
                stream=True,
            )
            texts = [""] * 2
            for chunk in batch:
                assert len(chunk.choices) == 1
                choice = chunk.choices[0]
                texts[choice.index] += choice.text
            results.append({
                "test": "streaming",
                "texts": texts,
            })

    n = len(results) // 2
    pp_results = results[:n]
    tp_results = results[n:]
    for pp, tp in zip(pp_results, tp_results):
        assert pp == tp
