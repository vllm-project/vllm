#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 SOURCE@sha256:DIGEST DESTINATION-zstd" >&2
  exit 2
fi

SOURCE=$1
DESTINATION=$2
if [[ ! "$SOURCE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[a-f0-9]{64}$ ]] ||
  [[ ! "$DESTINATION" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*:[a-zA-Z0-9_][a-zA-Z0-9_.-]*-zstd$ ]]; then
  echo "ERROR: source must use an immutable sha256 digest and destination must end in -zstd" >&2
  exit 2
fi

BUILDER=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "$BUILDER" ] && ! docker buildx rm --force "$BUILDER"; then
    echo "ERROR: failed to remove Buildx builder $BUILDER" >&2
    [ "$status" -ne 0 ] || status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

BUILDER=$(docker buildx create \
  --driver docker-container \
  --driver-opt network=host \
  --driver-opt 'image=moby/buildkit:v0.32.2@sha256:28a898719c18a33f4e8000685287fa36fd0dd9560c6440227d3a732d79bb41d8')

printf '# check=error=true\n\nFROM %s\n' "$SOURCE" | docker buildx build \
  --builder "$BUILDER" \
  --platform linux/amd64 \
  --progress plain \
  --provenance=mode=min \
  --output "type=image,name=${DESTINATION},push=true,compression=zstd,compression-level=3,force-compression=true,oci-mediatypes=true" \
  -
