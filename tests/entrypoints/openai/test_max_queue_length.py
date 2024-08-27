import asyncio
import pytest
from openai import InternalServerError
from ...utils import RemoteOpenAIServer

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["facebook/opt-125m"])
@pytest.mark.parametrize("max_queue_len", [1, 3, 5])
@pytest.mark.parametrize("max_num_seqs", [1, 2])
async def test_max_queue_length(model_name: str, max_queue_len: int,
                                max_num_seqs: int):

    print(f"\n \n--- Test Specs ---\n"
          f" model: {model_name}\n"
          f" max_queue_len: {max_queue_len}\n"
          f" max_num_seqs: {max_num_seqs}\n")

    server_args = [
        "--dtype",
        "half",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.2",
        "--max-queue-length",
        str(max_queue_len),
        "--max-num-seqs",
        str(max_num_seqs),
    ]

    with RemoteOpenAIServer(model_name, server_args) as server:

        client = server.get_async_client()
        client.max_retries = 0

        sample_prompts = [
            "Who won the world series in 2020?",
            "Where was the 2020 world series played?",
            "How long did the 2020 world series last?",
            "What were some television viewership statistics?",
            "Why was the 2020 world series so popular?"
        ]

        coroutines = [
            client.completions.create(
                prompt=sample_prompt,
                model=model_name,
                temperature=0.8,
                presence_penalty=0.2,
                max_tokens=400,
            ) for sample_prompt in sample_prompts
        ]

        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        actual_err_cnt = 0
        for i in range(len(responses)):
            if isinstance(responses[i], InternalServerError):
                assert responses[i].__dict__["code"] == 503
                actual_err_cnt += 1

        # Ensure that the number of err requests equals
        # the maximum between 0, and the number of
        # requests - max queue len - run queue len
        # where "-" is a minus sign
        expected_err_cnt = max(
            0,
            len(sample_prompts) - max_queue_len - max_num_seqs)
        print("Expected number of errors: ", expected_err_cnt)
        print("Actual number of errors: ", actual_err_cnt)
        assert expected_err_cnt == actual_err_cnt