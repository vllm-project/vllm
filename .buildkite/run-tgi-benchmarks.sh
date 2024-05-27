set -ex
set -o pipefail

# To run this script on your own, run
# `docker run -it --runtime nvidia --gpus all -v ./benchmark_containers:/benchmark_containers -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=<secret>"     --entrypoint /bin/bash     vllm/vllm-openai:latest`
# Then, open an empty bash script
# `vim run.sh`
# and paste the content of this file into `run.sh`
# Then `bash run.sh`

# install nvcc, in order to build tgi.
apt update
apt install -y wget curl apt-transport-https ca-certificates curl software-properties-common
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
apt-get update
apt-get install -y cuda-toolkit-12-1


# install conda
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
# equivalent to `source ~/.bashrc` but is runnable in non-interactive shell
eval "$(cat ~/.bashrc | tail -n +15)"


# create conda environment for tgi
conda create -n text-generation-inference python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate text-generation-inference

# Install tgi
(which cargo) || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source "$HOME/.cargo/env")
source "$HOME/.cargo/env" 
(which protoc) || (apt install protobuf-compiler -y)
(which pkg-config) || (apt install pkg-config -y)
apt install libssl-dev gcc -y
git clone https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference
BUILD_EXTENSIONS=True make install
# Install jmespath
pip install jmespath
# Install kernels needed by Llama-2 7B model
cd server
make -f Makefile-flash-att install-flash-attention
make -f Makefile-flash-att-v2 install-flash-attention-v2-cuda
# make -f Makefile-vllm install-vllm-cuda does not work. Use pip install workaround.
make -f Makefile-vllm build-vllm-cuda
cd vllm
pip install . -vvv
text-generation-launcher --port 8000 --model-id meta-llama/Llama-2-7b-chat-hf  &
tgi_pid=$!
echo $tgi_pid

# fall back to vllm python environment
conda deactivate
conda deactivate
conda deactivate

# prepare for benchmark
cd /vllm-workspace
git clone https://github.com/vllm-project/vllm.git
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


# gradually reduce the request rate from 20, untill all request successed
request_rate=20
get_successful_requests() {
    grep "Successful requests:" benchmark_serving.txt | awk '{print $3}'
}
# Loop until the successful requests are 1000
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


echo "### TGI Serving Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_serving.txt >> benchmark_results.md # first line
echo "" >> benchmark_results.md
echo '```' >> benchmark_results.md
tail -n 20 benchmark_serving.txt >> benchmark_results.md # last 20 lines
echo '```' >> benchmark_results.md

# if the agent binary is not found, skip uploading the results, exit 0
if [ ! -f /workspace/buildkite-agent ]; then
    exit 0
fi

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md
