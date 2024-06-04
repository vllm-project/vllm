#!/bin/bash

# This script should be run inside the vllm container. Enter the latest vllm container by
# docker run -it --runtime nvidia --gpus all --env "HF_TOKEN=<your HF TOKEN>"     --entrypoint /bin/bash  vllm/vllm-openai:latest
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into the docker and execute it using bash.
# Benchmarking results will be inside /vllm-workspace/vllm/benchmarks/*.txt

set -xe
set -o pipefail

# get the number of GPUs
gpu_count=$(nvidia-smi --list-gpus | wc -l)
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

if [[ $gpu_count -gt 0 ]]; then
  echo "GPU found."
else
  echo "Need at least 1 GPU to run benchmarking."
  exit 1
fi


# check if HF_TOKEN exists and starts with "hf_"
if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
  echo "Error: HF_TOKEN does not start with 'hf_'."
  exit 1
else
  echo "HF_TOKEN is set and valid."
fi

# install wget and curl
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
cd /
# git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/KuntaiDu/vllm.git
git checkout kuntai_benchmark
cd vllm/benchmarks/
mkdir results
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
PARAMS_FILE=/vllm/.buildkite/benchmark-parameters.json
RESULTS_FOLDER=/vllm/benchmarks/results/


# iterate over the test cases
jq -c '.[]' $PARAMS_FILE | while read -r params; do
    # extract keys
    keys=$(echo $params | jq -r 'keys_unsorted[]')

    # extract some parameters
    testname=$(echo $params | jq -r '.testname')
    model=$(echo $params | jq -r '.model')
    tp=$(echo $params | jq -r '.tensor-parallel-size')

    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
      continue
    fi

    # initialize the command with the script name
    offline_command="python3 benchmark_throughput.py --output-json $RESULTS_FOLDER/offline_$testname.json "
    online_command="python3 benchmark_serving.py --backend vllm --save-result --result-dir $RESULTS_FOLDER "
    
    # iteratre over each key to dynamically create variables and build the command
    for key in $keys; do
        value=$(echo $params | jq -r --arg key "$key" '.[$key]')
        if [[ key -eq "testname" ]]; then
          continue
        fi
        offline_command="$offline_command --$key $value"
        online_command="$online_command --$key $value"
    done
    
    # offline inference
    echo "Testing offline inference throughput ($testname)"
    sleep 5
    eval $offline_command 2>&1 | tee $RESULTS_FOLDER/offline_$testname.txt

    echo "### Offline inference throughput ($testname)" >> benchmark_results.md
    sed -n '1p' RESULTS_FOLDER/offline_$testname.txt >> benchmark_results.md # first line
    echo "" >> benchmark_results.md
    sed -n '$p' RESULTS_FOLDER/offline_$testname.txt >> benchmark_results.md # last line

    # online serving
    echo "Testing online serving throughput ($testname)"
    python3 -m vllm.entrypoints.openai.api_server --model $model --swap-space 16 --disable-log-requests -tp $tp &
    server_pid=$!
    # wait until server finishes initialization
    timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
    eval "$online_command" 2>&1 | tee $RESULTS_FOLDER/online_$testname.txt
    kill $server_pid
    # get the output json file and rename it
    serving_json_output=$(find "$RESULTS_FOLDER" -type f -name "vllm*")
    mv "$serving_json_output" $RESULTS_FOLDER/online_$testname.json

    echo "### Online serving throughput ($testname)" >> benchmark_results.md
    sed -n '1p' benchmark_serving.txt >> benchmark_results.md # first line
    echo "" >> benchmark_results.md
    echo '```' >> benchmark_results.md
    tail -n 17 benchmark_serving.txt >> benchmark_results.md # last 20 lines
    echo '```' >> benchmark_results.md




done


