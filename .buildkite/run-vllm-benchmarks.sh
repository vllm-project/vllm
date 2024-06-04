#!/bin/bash

# This script should be run inside the vllm container. Enter the latest vllm container by
# docker run -it --runtime nvidia --gpus all --env "HF_TOKEN=<your HF TOKEN>"  --entrypoint /bin/bash  vllm/vllm-openai:latest
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into the docker and execute it using bash.
# Benchmarking results will be at /vllm/benchmarks/results/benchmark_results.md

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
cd vllm
git checkout kuntai-tgibench-dev
cd benchmarks
mkdir results
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
PARAMS_FILE=/vllm/.buildkite/benchmark-parameters.json
RESULTS_FOLDER=/vllm/benchmarks/results/


(which jq) || (apt-get update && apt-get -y install jq)
# iterate over the test cases
jq -c '.[]' $PARAMS_FILE | while read -r params; do
    # extract keys
    keys=$(echo $params | jq -r 'keys_unsorted[]')

    # extract some parameters
    testname=$(echo $params | jq -r '.testname')
    model=$(echo $params | jq -r '.model')
    tp=$(echo $params | jq -r '.["tensor-parallel-size"]')

    if [[ $gpu_count -lt $tp ]]; then
      echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
      continue
    fi

    # initialize the command with the script name
    offline_command="python3 benchmark_throughput.py --output-json $RESULTS_FOLDER/offline_$testname.json "
    online_client_command="python3 benchmark_serving.py --backend vllm --save-result --result-dir $RESULTS_FOLDER "
    
    # iteratre over each key to dynamically create variables and build the command
    for key in $keys; do
        echo $key
        value=$(echo $params | jq -r ".[\"$key\"]")

        if [[ $key == "testname" ]]; then
          continue
        fi
        offline_command="$offline_command --$key $value"
        
        if [[ $key == "tensor-parallel-size" ]]; then
          # tensor-parallel-size is the server argument not the client.
          continue
        fi
        online_client_command="$online_client_command --$key $value"
        
        echo $online_client_command
    done
    
    # offline inference
    echo "Testing offline inference throughput ($testname)"
    eval $offline_command 2>&1 | tee $RESULTS_FOLDER/offline_$testname.txt

    echo "### Offline inference throughput ($testname)" >> $RESULTS_FOLDER/benchmark_results.md
    sed -n '1p' $RESULTS_FOLDER/offline_$testname.txt >> $RESULTS_FOLDER/benchmark_results.md # first line
    echo "" >> $RESULTS_FOLDER/benchmark_results.md
    sed -n '$p' $RESULTS_FOLDER/offline_$testname.txt >> $RESULTS_FOLDER/benchmark_results.md # last line

    # online serving
    echo "Testing online serving throughput ($testname)"
    # launch the server
    python3 -m vllm.entrypoints.openai.api_server --model $model --swap-space 16 --disable-log-requests -tp $tp &
    server_pid=$!
    # wait until server finishes initialization
    timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
    eval "$online_client_command" 2>&1 | tee $RESULTS_FOLDER/online_$testname.txt
    kill $server_pid
    # get the output json file and rename it
    serving_json_output=$(find "$RESULTS_FOLDER" -type f -name "vllm*")
    mv "$serving_json_output" $RESULTS_FOLDER/online_$testname.json


    # document the results
    echo "### Online serving throughput ($testname)" >> $RESULTS_FOLDER/benchmark_results.md
    sed -n '1p' $RESULTS_FOLDER/online_$testname.txt >> $RESULTS_FOLDER/benchmark_results.md # first line
    echo "" >> $RESULTS_FOLDER/benchmark_results.md
    echo '```' >> $RESULTS_FOLDER/benchmark_results.md
    tail -n 17 $RESULTS_FOLDER/online_$testname.txt >> $RESULTS_FOLDER/benchmark_results.md # last 20 lines
    echo '```' >> $RESULTS_FOLDER/benchmark_results.md

done

# if the agent binary is not found, skip uploading the results, exit 0
if [ ! -f /workspace/buildkite-agent ]; then
    exit 0
fi

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < $RESULTS_FOLDER/benchmark_results.md

# upload artifacts
/workspace/buildkite-agent artifact upload $RESULTS_FOLDER/*
