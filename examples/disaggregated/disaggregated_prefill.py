# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""

import os
import time
from multiprocessing import Event, Process
from uuid import uuid4

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.sampling_params import RequestOutputKind
from vllm.utils.network_utils import get_ip

MODEL = os.environ.get(
    "VLLM_DISAGG_PREFILL_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
PREFILL_KV_PORT = 14579
DECODE_KV_PORT = 14580


def _build_request_ids(num_requests: int, host: str) -> list[str]:
    prefill_addr = f"{host}:{PREFILL_KV_PORT}"
    decode_addr = f"{host}:{DECODE_KV_PORT}"
    return [
        (
            f"___prefill_addr_{prefill_addr}"
            f"___decode_addr_{decode_addr}_{uuid4().hex}"
        )
        for _ in range(num_requests)
    ]


def _generate_with_request_ids(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    request_ids: list[str],
):
    if len(prompts) != len(request_ids):
        raise ValueError("prompts and request_ids must have the same length")

    params = sampling_params.clone()
    params.output_kind = RequestOutputKind.FINAL_ONLY

    request_order = {
        request_id: index for index, request_id in enumerate(request_ids)
    }
    internal_request_ids: list[str] = []

    try:
        for request_id, prompt in zip(request_ids, prompts):
            prompt_input = llm._preprocess_cmpl_one(prompt)
            internal_request_ids.append(
                llm.llm_engine.add_request(request_id, prompt_input, params)
            )

        outputs = []
        while llm.llm_engine.has_unfinished_requests():
            outputs.extend(
                output
                for output in llm.llm_engine.step()
                if output.finished
            )

        return sorted(outputs, key=lambda output: request_order[output.request_id])
    except Exception:
        if internal_request_ids:
            llm.llm_engine.abort_request(internal_request_ids, internal=True)
        raise


def run_prefill(prefill_done, request_ids):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        # The decode node will actually "prefill" this request.
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_port=PREFILL_KV_PORT,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model=MODEL,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    _generate_with_request_ids(llm, prompts, sampling_params, request_ids)
    print("Prefill node is finished.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done, request_ids):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_port=DECODE_KV_PORT,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model=MODEL,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    # Wait for the producer to start the pipe
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    outputs = _generate_with_request_ids(llm, prompts, sampling_params, request_ids)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    request_ids = _build_request_ids(num_requests=3, host=get_ip())
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, request_ids))
    decode_process = Process(target=run_decode, args=(prefill_done, request_ids))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node when decode is finished
    decode_process.join()
    prefill_process.terminate()


if __name__ == "__main__":
    main()
