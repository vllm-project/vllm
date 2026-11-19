#!/usr/bin/env bash
set -euo pipefail
staging=public.ecr.aws/q9t5s3a7/vllm-release-repo
digest=$(buildkite-agent meta-data get rust-frontend-x86_64)
[[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
frontend="$staging@$digest"
engine="$staging:$BUILDKITE_COMMIT-x86_64"
model=Qwen/Qwen3-0.6B
revision=c1899de289a04d12100db370d81485cdf75e47ca
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${staging%/*}"
docker pull "$engine"
docker pull "$frontend"
frontend_version=$(docker run --rm "$frontend" --version)
engine_version=$(docker run --rm --entrypoint vllm-rs "$engine" --version)
test -n "$frontend_version"
test "$frontend_version" = "$engine_version"

server=
worker=
cleanup() {
    for id in "$worker" "$server"; do
        if [[ -n $id ]]; then
            docker logs --tail 100 "$id" >&2
            docker rm -f "$id" >/dev/null
        fi
    done
}
trap cleanup EXIT
dp=(--data-parallel-address 127.0.0.1 --data-parallel-rpc-port 62100 --data-parallel-size 1)
server=$(docker run -d -p 127.0.0.1::8000 "$frontend" serve "$model" \
    --revision "$revision" --host 0.0.0.0 --max-model-len 256 "${dp[@]}" --data-parallel-size-local 0)
worker=$(docker run -d --gpus all --shm-size 2g --network "container:$server" \
    --entrypoint vllm "$engine" serve "$model" --revision "$revision" --headless \
    --enforce-eager --max-model-len 256 "${dp[@]}" --data-parallel-size-local 1)
url="http://$(docker port "$server" 8000/tcp)"
curl -fsS --retry 180 --retry-all-errors --retry-delay 2 --max-time 5 "$url/health"
curl -fsS --max-time 60 "$url/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":1,"temperature":0,
         "allowed_token_ids":[42],"return_token_ids":true}' | jq -e '.choices[0].token_ids == [42]'
