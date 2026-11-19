#!/usr/bin/env bash
# Run on the native worker for each release platform.
set -euo pipefail
repo=public.ecr.aws/q9t5s3a7/vllm-release-repo
arch=$(uname -m)
image="$repo:$BUILDKITE_COMMIT-$arch-rust-frontend"

if ! command -v uv >/dev/null; then
    curl -fsSL https://astral.sh/uv/0.13.4/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
version=$(SETUPTOOLS_SCM_PRETEND_METADATA='{dirty=false}' uv run --no-project \
    --with-requirements requirements/build/rust.txt --python 3.12 python -c \
    'from tools.build_rust import prepare_build_environment; v=prepare_build_environment(); assert v; print(v)')
# The shared helper emits --build-arg and --tag arguments for Docker.
read -r -a metadata <<< "$(bash .buildkite/scripts/docker-build-metadata-args.sh rust-frontend)"
docker buildx build --load --provenance=false "${metadata[@]}" --build-arg "VLLM_RS_BUILD_VERSION=$version" \
    -f docker/Dockerfile.rust .
test "$(docker run --rm "$image" --version | awk '{print $NF}')" = "$version"

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${repo%/*}"
docker push "$image"
digest=$(docker image inspect "$image" --format '{{json .RepoDigests}}' | \
    jq -er --arg repo "$repo@" '.[] | select(startswith($repo)) | split("@")[1]')
[[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
buildkite-agent meta-data set "rust-frontend-$arch" "$digest"
container=$(docker create "$image")
trap 'docker rm "$container" >/dev/null' EXIT
printf 'Uncompressed root filesystem tar bytes: '
docker export "$container" | wc -c
printf 'Compressed layer bytes: '
docker buildx imagetools inspect "$repo@$digest" --raw | jq -e '[.layers[].size] | add'
