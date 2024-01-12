# This script is run by buildkite to run the benchmarks and upload the results to buildkite

set -ex

# cd into parent directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# run benchmarks and upload the result to buildkite
python3 benchmarks/benchmark_latency.py 2>&1 | tee benchmark_latency.txt

python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 2>&1 | tee benchmark_throughput.txt

# write the results into a markdown file
echo "### Latency Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_latency.txt >> benchmark_results.md
echo "" >> benchmark_results.md
sed -n '$p' benchmark_latency.txt >> benchmark_results.md
echo "### Throughput Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_throughput.txt >> benchmark_results.md
echo "" >> benchmark_results.md
sed -n '$p' benchmark_throughput.txt >> benchmark_results.md

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md
