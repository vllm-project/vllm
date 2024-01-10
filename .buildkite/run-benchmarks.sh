set -ex

# cd into parent directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# run benchmarks and upload the result to buildkite
python3 benchmarks/benchmark_latency.py 2>&1 | tee benchmark_latency.txt

python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 2>&1 | tee benchmark_throughput.txt

# write the results into a markdown file
cat << EOF > benchmark_results.md
# Latency
${cat benchmark_latency.txt}

# Throughput
${cat benchmark_throughput.txt}
EOF

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md