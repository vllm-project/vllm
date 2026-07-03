#!/bin/bash

set -euo pipefail

test_suite="${1:-}"

if [[ -z "${test_suite}" ]]; then
  echo "Usage: $0 <example|v1|server>" >&2
  exit 1
fi

case "${test_suite}" in
  example)
    pip install tblib==3.1.0

    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 -O3 -cc.cudagraph_mode=NONE
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager -tp 2 --distributed-executor-backend mp
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager --attention-backend=TRITON_ATTN
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager --quantization fp8
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --block-size 64 --enforce-eager --kv-cache-dtype fp8
    python3 examples/basic/offline_inference/generate.py --model nvidia/Llama-3.1-8B-Instruct-FP8 --block-size 64 --enforce-eager --quantization modelopt --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 4096
    python3 examples/basic/offline_inference/generate.py --model superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4 --block-size 64 --enforce-eager --max-model-len 8192
    python3 examples/basic/offline_inference/generate.py --model TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ --block-size 64 --enforce-eager 
    python3 examples/basic/offline_inference/generate.py --model ibm-research/PowerMoE-3b --block-size 64 --enforce-eager -tp 2
    python3 examples/basic/offline_inference/generate.py --model ibm-research/PowerMoE-3b --block-size 64 --enforce-eager -tp 2 --enable-expert-parallel
    python3 examples/basic/offline_inference/generate.py --model superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4 --max-model-len 8192
    ;;
  v1)
    cd tests

    pytest -v -s v1/core --ignore=v1/core/test_reset_prefix_cache_e2e.py --ignore=v1/core/test_scheduler_e2e.py
    pytest -v -s v1/engine --ignore=v1/engine/test_output_processor.py
    pytest -v -s v1/sample --ignore=v1/sample/test_logprobs.py --ignore=v1/sample/test_logprobs_e2e.py -k "not test_topk_only and not test_topp_only and not test_topk_and_topp"
    pytest -v -s v1/worker --ignore=v1/worker/test_gpu_model_runner.py --ignore=v1/worker/test_worker_memory_snapshot.py
    pytest -v -s v1/structured_output
    pytest -v -s v1/test_serial_utils.py
    pytest -v -s v1/spec_decode --ignore=v1/spec_decode/test_max_len.py --ignore=v1/spec_decode/test_speculators_eagle3.py --ignore=v1/spec_decode/test_acceptance_length.py --ignore=v1/spec_decode/test_speculators_correctness.py
    pytest -v -s v1/kv_connector/unit --ignore=v1/kv_connector/unit/test_multi_connector.py --ignore=v1/kv_connector/unit/test_example_connector.py --ignore=v1/kv_connector/unit/test_lmcache_integration.py --ignore=v1/kv_connector/unit/test_hf3fs_client.py --ignore=v1/kv_connector/unit/test_hf3fs_connector.py --ignore=v1/kv_connector/unit/test_hf3fs_metadata_server.py --ignore=v1/kv_connector/unit/test_offloading_connector.py
    ;;
  server)
    pip install av
    cd tests

    pytest -v -s entrypoints/multimodal/openai/chat_completion/test_audio_in_video.py
    pytest -v -s benchmarks/test_serve_cli.py
    ;;
  *)
    echo "Unknown Intel test suite: ${test_suite}" >&2
    exit 1
    ;;
esac
