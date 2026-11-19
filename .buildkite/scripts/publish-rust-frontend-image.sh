#!/usr/bin/env bash
set -euo pipefail
staging=${VLLM_STAGING_IMAGE_REPO:-public.ecr.aws/q9t5s3a7/vllm-release-repo}
hub=vllm/vllm-rust-frontend

manifest_digest() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest.Digest}}' | jq -er .
}

sources=()
for arch in x86_64 aarch64; do
    digest=$(buildkite-agent meta-data get "rust-frontend-$arch")
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "Invalid staged digest for $arch" >&2; exit 1; }
    sources+=("$staging@$digest")
done
case "${1:-}" in
    nightly) tag="nightly-$BUILDKITE_COMMIT"; moving=nightly ;;
    release)
        version=$(buildkite-agent meta-data get release-version)
        tag="v${version#v}"
        moving=latest ;;
    *) echo "Usage: $0 {nightly|release}" >&2; exit 2 ;;
esac

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${staging%/*}"
index="$staging:$BUILDKITE_BUILD_ID-$moving-rust-frontend"
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
    # Release candidates get a versioned tag only.
    [[ $tag =~ ^v[0-9]+(\.[0-9]+){2,}(\.post[0-9]+)?$ ]] || exit 0
fi
docker buildx imagetools create -t "$hub:$moving" -t "vllm/vllm:$moving-rust-frontend" "$hub@$digest"
