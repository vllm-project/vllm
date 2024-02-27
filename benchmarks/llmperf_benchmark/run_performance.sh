#!/bin/bash
trap_ctrlc() {
	kill $server_pid
}
set -uexo pipefail
command -v curl &> /dev/null
if [ $? != 0 ]; then
	apt-get update && apt-get install -y curl
fi
command -v python3 -c 'import llmperf' &>/dev/null
if [ $? != 0 ]; then
	pip3 install -e .
fi
workdir="$(realpath $(dirname $0))"

cd $workdir/llmperf

# start server
python3 -m vllm.entrypoints.openai.api_server --model meta-llama/llama-2-7b-chat-hf &
server_pid=$!
trap trap_ctrlc INT

timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:8000/v1"
python token_benchmark_ray.py \
	--model "meta-llama/Llama-2-7b-chat-hf" \
	--mean-input-tokens 550 \
	--stddev-input-tokens 150 \
	--mean-output-tokens 150 \
	--stddev-output-tokens 10 \
	--max-num-completed-requests 2 \
	--timeout 600 \
	--num-concurrent-requests 1 \
	--results-dir "result_outputs" \
	--llm-api openai \
	--additional-sampling-params '{}'

kill $server_pid

