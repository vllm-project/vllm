import os

from ..utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
EAGER_MODE = bool(int(os.getenv("EAGER_MODE", 0)))
CHUNKED_PREFILL = bool(int(os.getenv("CHUNKED_PREFILL", 0)))
TP_SIZE = int(os.getenv("TP_SIZE", 1))
PP_SIZE = int(os.getenv("PP_SIZE", 1))


def test_compare_tp():
    pp_args = [
        "--model",
        MODEL_NAME,
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
    tp_args = [
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(TP_SIZE),
    ]
    if CHUNKED_PREFILL:
        pp_args.append("--enable-chunked-prefill")
        tp_args.append("--enable-chunked-prefill")
    if EAGER_MODE:
        pp_args.append("--enforce-eager")
        tp_args.append("--enforce-eager")

    results = []
    for args in [pp_args, tp_args]:
        with RemoteOpenAIServer(args) as server:
            client = server.get_client()

            # test models list
            models = client.models.list()
            results.append(models)

            # test with text prompt
            completion = client.completions.create(model=MODEL_NAME,
                                                   prompt="Hello, my name is",
                                                   max_tokens=5,
                                                   temperature=0.0)

            results.append(completion)

            # test using token IDs
            completion = client.completions.create(
                model=MODEL_NAME,
                prompt=[0, 0, 0, 0, 0],
                max_tokens=5,
                temperature=0.0,
            )

            results.append(completion)

            # test simple list
            batch = client.completions.create(
                model=MODEL_NAME,
                prompt=["Hello, my name is", "Hello, my name is"],
                max_tokens=5,
                temperature=0.0,
            )

            results.append(batch)

            # test streaming
            batch = client.completions.create(
                model=MODEL_NAME,
                prompt=["Hello, my name is", "Hello, my name is"],
                max_tokens=5,
                temperature=0.0,
                stream=True,
            )
            batch = list(batch)
            results.append(batch)
    n = len(results) // 2
    pp_results = results[:n]
    tp_results = results[n:]
    for pp, tp in zip(pp_results, tp_results):
        print((pp, tp))
