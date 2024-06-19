#!/bin/bash
set -uxo pipefail

# we will need to download test models off HF hub
unset HF_HUB_OFFLINE

export HTTP_PORT=8080
export GRPC_PORT=8033


function wait_for(){
    trap "" ERR # we don't care about errors in this function

    name=$1
    shift
    command=$@

    max_retries=10
    until $command ; do
        echo "Waiting for $name to be up (retries_left=$max_retries)..."
        sleep 30
        max_retries=$((max_retries-1))
        if [[ max_retries -le 0 ]]; then
            echo "Timed out waiting for $name server" >&2
            exit 1
        fi
    done
}

# stop the server on any errors
trap 'kill -9 $server_pid && exit 1' ERR

# spin up the OpenAPI server in the background
python -m vllm.entrypoints.openai.api_server --port $HTTP_PORT &
server_pid=$!
server_url="http://localhost:$HTTP_PORT"

wait_for "http server" curl --verbose --connect-timeout 1 --fail-with-body --no-progress-meter "${server_url}/health"

curl -v --no-progress-meter --fail-with-body \
  "${server_url}/v1/models" | python -m json.tool || \

curl -v --no-progress-meter --fail-with-body \
  --header "Content-Type: application/json" \
  --data '{
    "prompt": "A red fedora symbolizes ",
    "model": "facebook/opt-125m"
}' \
  "${server_url}/v1/completions" | python -m json.tool

echo "OpenAI API success" && kill -9 $server_pid


# spin up the grpc server in the background
python -m vllm_tgis_adapter --grpc-port $GRPC_PORT &
server_pid=$!
server_url="localhost:$GRPC_PORT"

# get grpcurl
curl --no-progress-meter --location --output /tmp/grpcurl.tar.gz \
  https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz
tar -xf /tmp/grpcurl.tar.gz --directory /tmp

wait_for "grpc_server" grpc_healthcheck # healthcheck is part of vllm_tgis_adapter

/tmp/grpcurl -v \
    -plaintext \
    -use-reflection \
    -d '{ "requests": [{"text": "A red fedora symbolizes "}]}' \
    "$server_url" \
    fmaas.GenerationService/Generate

echo "GRPC API success" && kill -9 $server_pid
