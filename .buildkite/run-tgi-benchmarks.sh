#!/bin/bash

# This script should be run inside the vllm container. Enter the latest vllm container by
# docker run -it --gpus all -e "HF_TOKEN=<your HF TOKEN>"  --shm-size 1g --entrypoint /bin/bash ghcr.io/huggingface/text-generation-inference:2.0
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into the docker and execute it using bash.
# Benchmarking results will be inside /vllm/benchmarks/*.txt
# NOTE: this script gradually reduces the request rate from 20, to ensure all requests are successful.

set -ex
set -o pipefail

# install conda
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
eval "$(cat ~/.bashrc | tail -n +15)"

# create conda environment for vllm
conda create -n vllm python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate vllm
pip install vllm

# clone vllm repo
cd /
git clone https://github.com/vllm-project/vllm.git
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# launch TGI server
/tgi-entrypoint.sh --port 8000 --model-id meta-llama/Llama-2-7b-chat-hf &
tgi_pid=$!
timeout 600 bash -c 'until curl localhost:8000/generate_stream; do sleep 1; done' || exit 1

# gradually reduce the request rate from 20, untill all request successed
request_rate=20
get_successful_requests() {
  grep "Successful requests:" benchmark_serving.txt | awk '{print $3}'
}
while true; do
  echo "Running benchmark with request rate $request_rate..."
  python3 vllm/benchmarks/benchmark_serving.py --backend tgi --model meta-llama/Llama-2-7b-chat-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --endpoint /generate_stream --request-rate $request_rate --port 8000 --save-result 2>&1 | tee benchmark_serving.txt
  bench_serving_exit_code=$?
  successful_requests=$(get_successful_requests)
  echo "Successful requests: $successful_requests"
  if [ "$successful_requests" -eq 1000 ]; then
    echo "Reached 1000 successful requests with request rate $request_rate"
    break
  fi
  request_rate=$((request_rate - 1))
  if [ "$request_rate" -lt 1 ]; then
    echo "Request rate went below 1. Exiting."
    break
  fi
done
kill $tgi_pid

echo "### TGI Serving Benchmarks" >>benchmark_results.md
sed -n '1p' benchmark_serving.txt >>benchmark_results.md
echo "" >>benchmark_results.md
echo '```' >>benchmark_results.md
tail -n 17 benchmark_serving.txt >>benchmark_results.md
echo '```' >>benchmark_results.md

# if the agent binary is not found, skip uploading the results, exit 0
if [ ! -f /workspace/buildkite-agent ]; then
  exit 0
fi

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" <benchmark_results.md
