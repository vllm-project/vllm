# This script is run by buildkite to run the benchmarks and upload the results to buildkite

set -ex
set -o pipefail

# cd into parent directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(wget && curl) || (apt-get update && apt-get install -y wget curl)

# run benchmarks and upload the result to buildkite
kv_cache_dtypes=("auto" "fp8_e5m2")
for cache_dtype in "${kv_cache_dtypes[@]}"; do
    python3 benchmarks/benchmark_latency.py --kv-cache-dtype ${cache_dtype} 2>&1 | tee -a benchmark_latency.txt
    bench_latency_exit_code=$?

    python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --kv-cache-dtype ${cache_dtype} 2>&1 | tee -a benchmark_throughput.txt
    bench_throughput_exit_code=$?

    python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --kv-cache-dtype ${cache_dtype} &
    server_pid=$!

    dataset_file="ShareGPT_V3_unfiltered_cleaned_split.json"
    if [ ! -f "${dataset_file}" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/${dataset_file}
    fi
    # wait for server to start, timeout after 600 seconds
    timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
    python3 benchmarks/benchmark_serving.py \
        --dataset ./${dataset_file} \
        --model meta-llama/Llama-2-7b-chat-hf \
        --num-prompts 20 \
        --endpoint /v1/completions \
        --tokenizer meta-llama/Llama-2-7b-chat-hf 2>&1 | tee -a benchmark_serving.txt
    bench_serving_exit_code=$?
    kill $server_pid
done

# write the results into a markdown file
echo "### Latency Benchmarks" >> benchmark_results.md
sed -n '/Namespace/p' benchmark_latency.txt >> benchmark_results.md # config info
echo "" >> benchmark_results.md
sed -n '/latency:/p' benchmark_latency.txt >> benchmark_results.md # results

echo "### Throughput Benchmarks" >> benchmark_results.md
sed -n '/Namespace/p' benchmark_throughput.txt >> benchmark_results.md # config info
echo "" >> benchmark_results.md
sed -n '/Throughput:/p' benchmark_throughput.txt >> benchmark_results.md # results

echo "### Serving Benchmarks" >> benchmark_results.md
sed -n '/Namespace/p' benchmark_serving.txt >> benchmark_results.md # config info
echo "" >> benchmark_results.md
sed -n '/Total time:/,+4p' benchmark_serving.txt >> benchmark_results.md # last 5 lines

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md

# exit with the exit code of the benchmarks
if [ $bench_latency_exit_code -ne 0 ]; then
    exit $bench_latency_exit_code
fi

if [ $bench_throughput_exit_code -ne 0 ]; then
    exit $bench_throughput_exit_code
fi

if [ $bench_serving_exit_code -ne 0 ]; then
    exit $bench_serving_exit_code
fi
