#!/usr/bin/env bash
# Run on the native worker for each release platform.
set -euo pipefail
repo=${VLLM_STAGING_IMAGE_REPO:-public.ecr.aws/q9t5s3a7/vllm-release-repo}
arch=$(uname -m)

if ! command -v uv >/dev/null; then
    curl -fsSL https://astral.sh/uv/0.13.4/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
version=$(SETUPTOOLS_SCM_PRETEND_METADATA='{dirty=false}' uv run --no-project \
    --with-requirements requirements/build/rust.txt --python 3.12 python -c \
    'from tools.build_rust import prepare_build_environment; v=prepare_build_environment(); assert v; print(v)')
# The shared helper emits --build-arg and --tag arguments for Docker.
read -r -a metadata <<< "$(bash .buildkite/scripts/docker-build-metadata-args.sh rust-frontend)"
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${repo%/*}"
receipt=$(mktemp)
trap 'rm -f "$receipt"' EXIT
docker buildx build --push --provenance=false --metadata-file "$receipt" "${metadata[@]}" \
    --build-arg "VLLM_RS_BUILD_VERSION=$version" -f docker/Dockerfile.rust .
digest=$(jq -er '."containerimage.digest"' "$receipt")
[[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
test "$(docker run --rm "$repo@$digest" --version | awk '{print $NF}')" = "$version"
buildkite-agent meta-data set "rust-frontend-$arch" "$digest"
