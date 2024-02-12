# This script is run by buildkite to run the benchmarks and upload the results to buildkite

set -ex

# cd into parent directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# run python backend benchmarks and upload the result to buildkite
python3 benchmarks/benchmark_latency.py 2>&1 | tee benchmark_latency.txt

python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 2>&1 | tee benchmark_throughput.txt

# run serving benchmark and upload the result to buildkite
MODEL="facebook/opt-125m"

# start the server in a separate process (need to switch dir to launch vllm server as a module)
nohup sh -c "cd benchmarks && python3 -m vllm.entrypoints.api_server --model $MODEL --swap-space 16 --disable-log-requests" &

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

echo "Waiting for vLLM server to be ready..."
while :; do
  curl -s --fail -o /dev/null "http://localhost:8000/health" && break
  sleep 1 # just a little buffer
done

echo "Starting serving benchmark..."
python3 benchmarks/serving/benchmark_serving.py \
        --model $MODEL \
        --dataset "ShareGPT_V3_unfiltered_cleaned_split.json" \
        2>&1 | tee benchmark_serving.txt

# cleanup
pkill -9 python3

# write the results into a markdown file
echo "### Latency Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_latency.txt >> benchmark_results.md
echo "" >> benchmark_results.md
sed -n '$p' benchmark_latency.txt >> benchmark_results.md
echo "### Throughput Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_throughput.txt >> benchmark_results.md
echo "" >> benchmark_results.md
sed -n '$p' benchmark_throughput.txt >> benchmark_results.md
echo "### Serving Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_serving.txt >> benchmark_results.md
echo "" >> benchmark_results.md
tail -n 13 benchmark_serving.txt >> benchmark_results.md

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md
