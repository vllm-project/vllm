#!/usr/bin/env bash
set -euo pipefail
staging=public.ecr.aws/q9t5s3a7/vllm-release-repo
hub=vllm/vllm-rust-frontend

manifest_digest() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest.Digest}}' | jq -er .
}

sources=()
for arch in x86_64 aarch64; do
    digest=$(buildkite-agent meta-data get "rust-frontend-$arch")
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
    sources+=("$staging@$digest")
done
tag=${1:?expected nightly or a version tag}
if [[ $tag == nightly ]]; then
    tag="nightly-$BUILDKITE_COMMIT"
    moving=nightly
else
    moving=latest
fi

index="$staging:$BUILDKITE_COMMIT-rust-frontend"
docker buildx imagetools create -t "$index" "${sources[@]}"
digest=$(manifest_digest "$index")
index="$staging@$digest"
docker buildx imagetools inspect "$index" --raw | jq -e \
    --arg amd "${sources[0]##*@}" --arg arm "${sources[1]##*@}" \
    '([.manifests[] | [.platform.os, .platform.architecture, .digest]] | sort)
     == [["linux","amd64",$amd],["linux","arm64",$arm]]'

# A retry may repair an alias, but must never replace a published version.
if existing=$(manifest_digest "$hub:$tag" 2>&1); then
    if [[ $existing != "$digest" ]]; then
        echo "$hub:$tag already points to $existing, refusing to replace it" >&2
        exit 1
    fi
elif [[ $existing == *"not found"* || $existing == *"manifest unknown"* ]]; then
    docker buildx imagetools create -t "$hub:$tag" "$index"
else
    echo "$existing" >&2
    exit 1
fi
test "$(manifest_digest "$hub:$tag")" = "$digest"
if [[ $moving == latest ]]; then
    docker buildx imagetools create -t "vllm/vllm:$tag-rust-frontend" "$hub@$digest"
    # A patch on an older release branch must not move the current stable tag.
    newest=$( { git tag -l 'v[0-9]*'; echo "$tag"; } | grep -Ex 'v[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -1)
    [[ $newest == "$tag" ]] || exit 0
else
    # A late retry of an older nightly must not roll the frontend back.
    engine_pinned=$(manifest_digest "vllm/vllm-openai:$tag")
    engine_current=$(manifest_digest vllm/vllm-openai:nightly)
    [[ $engine_pinned == "$engine_current" ]] || exit 0
fi
docker buildx imagetools create -t "$hub:$moving" -t "vllm/vllm:$moving-rust-frontend" "$hub@$digest"
