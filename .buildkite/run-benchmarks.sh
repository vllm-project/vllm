set -ex

# run benchmarks and upload the result to buildkite
python benchmarks/benchmark_latency.py 2>&1 | tee benchmark_latency.txt
python benchmarks/benchmark_throughput.py 2>&1 | tee benchmark_throughput.txt

python -m vllm.entrypoints.api_server &
API_SERVER_PID=$!
python benchmarks/benchmark_serving.py 2>&1 | tee benchmark_serving.txt
kill $API_SERVER_PID

# write the results into a markdown file
cat << EOF > benchmark_results.md
# Latency
${cat benchmark_latency.txt}

# Throughput
${cat benchmark_throughput.txt}

# Serving
${cat benchmark_serving.txt}
EOF

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md