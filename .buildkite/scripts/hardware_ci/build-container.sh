#!/bin/sh
set -ex

# ---------------------------------------------------------------------------
# Build the vllm XPU container image using the in-cluster BuildKit daemon
# and push it to the registry. No git clone needed — Buildkite has already
# checked out the repo into the workspace at /buildkite/workspace.
#
# Required env vars (set by the pipeline):
#   REGISTRY        — registry hostname, e.g. quay.io
#   REGISTRY_USER   — registry username
#   REGISTRY_TOKEN  — registry password/token (injected as a secret)
#   BUILDKITE_COMMIT — set automatically by Buildkite
# ---------------------------------------------------------------------------

IMAGE_NAME="${REGISTRY}/${REGISTRY_USER}/vllm-ci:${BUILDKITE_COMMIT}"
BUILDKIT_ADDR="tcp://buildkit.default.svc.cluster.local:1234"
DOCKERFILE="${BUILDKITE_BUILD_CHECKOUT_PATH}/docker/Dockerfile.xpu"
CONTEXT="${BUILDKITE_BUILD_CHECKOUT_PATH}"

# Set up registry auth so buildctl can push.
# buildctl uses the standard Docker config at $DOCKER_CONFIG/config.json.
DOCKER_CONFIG_DIR="${HOME}/.docker"
mkdir -p "${DOCKER_CONFIG_DIR}"
AUTH=$(printf '%s:%s' "${REGISTRY_USER}" "${REGISTRY_TOKEN}" | base64 | tr -d '\n')
cat > "${DOCKER_CONFIG_DIR}/config.json" <<EOF
{
  "auths": {
    "${REGISTRY}": {
      "auth": "${AUTH}"
    }
  }
}
EOF

echo "Building image: ${IMAGE_NAME}"
echo "Context:        ${CONTEXT}"
echo "Dockerfile:     ${DOCKERFILE}"

buildctl \
  --addr "${BUILDKIT_ADDR}" \
  build \
  --frontend dockerfile.v0 \
  --local context="${CONTEXT}" \
  --local dockerfile="${CONTEXT}/docker" \
  --opt filename=Dockerfile.tmp \
  --output "type=image,name=${IMAGE_NAME},push=true"

echo "Build complete: ${IMAGE_NAME}"
