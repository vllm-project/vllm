#!/bin/bash
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-xpu"
kernel_nightly="${VLLM_XPU_KERNEL_NIGHTLY:-0}"
case "$kernel_nightly" in
  0|1) ;;
  *)
    echo "VLLM_XPU_KERNEL_NIGHTLY must be 0 or 1, got: $kernel_nightly" >&2
    exit 1
    ;;
esac

if [[ "$kernel_nightly" == "1" ]]; then
  : "${BUILDKITE_JOB_ID:?Kernel nightly requires a Buildkite job}"
  # Invalidate a previous successful attempt before retrying this step.
  buildkite-agent meta-data set xpu-kernel-nightly-image pending
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null; then
    export UV_INSTALL_DIR="$HOME/.local/bin"
    curl --fail --show-error --silent --location --retry 3 \
      https://astral.sh/uv/0.11.1/install.sh | sh
  fi
  kernel_dir="$(mktemp -d)"
  cleanup_kernel_nightly() {
    status=$?
    if [[ -f "$kernel_dir/xpu.txt" ]]; then
      cp "$kernel_dir/xpu.txt" requirements/xpu.txt
    fi
    rm -f "$kernel_dir/report.json" "$kernel_dir/selection.txt" "$kernel_dir/xpu.txt"
    rmdir "$kernel_dir"
    return "$status"
  }
  trap cleanup_kernel_nightly EXIT
  uv run --isolated --no-project --python 3.12 --with pip --with packaging \
    python - "$kernel_dir" "${VLLM_XPU_KERNEL_VERSION:-}" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.utils import parse_wheel_filename
from packaging.version import Version

directory = Path(sys.argv[1])
requested = sys.argv[2]
requirement = "vllm-xpu-kernels"
if requested:
    if not Version(requested).is_devrelease:
        sys.exit("VLLM_XPU_KERNEL_VERSION must be a development version")
    requirement += f"=={requested}"
subprocess.run(
    [sys.executable, "-m", "pip", "--isolated", "install", "--dry-run",
     "--ignore-installed", "--pre", "--no-deps", "--only-binary=:all:",
     "--index-url",
     "https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/",
     "--report", str(directory / "report.json"), requirement],
    check=True,
)
report = json.loads((directory / "report.json").read_text())
items = report["install"]
if len(items) != 1:
    sys.exit("Expected exactly one kernel wheel")
item = items[0]
url = item["download_info"]["url"]
wheel = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
name, version, _, _ = parse_wheel_filename(wheel)
sha256 = item["download_info"]["archive_info"]["hashes"]["sha256"]
if (name != "vllm-xpu-kernels" or not version.is_devrelease
        or str(version) != item["metadata"]["version"]
        or urlsplit(url).scheme != "https"
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)):
    sys.exit("Invalid kernel nightly wheel metadata")
commit = re.search(r"(?:^|\.)g([0-9a-f]+)(?:\.|$)", version.local or "")
if not commit:
    sys.exit("Kernel nightly version is missing its commit")
(directory / "selection.txt").write_text(
    "\n".join([wheel, str(version), url, sha256, commit[1]]) + "\n"
)
PY
  mapfile -t selection < "$kernel_dir/selection.txt"
  wheel_name="${selection[0]}"
  kernel_version="${selection[1]}"
  wheel_url="${selection[2]}"
  wheel_sha256="${selection[3]}"
  kernel_sha="${selection[4]}"
  IMAGE="$REGISTRY/$REPO:${BUILDKITE_COMMIT}-kernel-${kernel_sha}-${BUILDKITE_JOB_ID}-xpu"
  mkdir -p build/xpu-kernel-wheel
  info_file="build/xpu-kernel-wheel/kernel-wheel-info.txt"
  printf 'wheel=%s\nversion=%s\nurl=%s\nsha256=%s\nimage=%s\n' \
    "$wheel_name" "$kernel_version" "$wheel_url" "$wheel_sha256" "$IMAGE" \
    | tee "$info_file"
  buildkite-agent artifact upload "$info_file"
  annotate_kernel_wheel() {
    printf "### XPU Kernel Nightly: %s\n\nWheel: \`%s\`\n\nVersion: \`%s\`\n\n[Download wheel](%s)\n\nSHA256: \`%s\`\n\nImage: \`%s\`\n" \
      "$1" "$wheel_name" "$kernel_version" "$wheel_url" "$wheel_sha256" "$IMAGE" \
      | buildkite-agent annotate --style info --context "kernel-nightly-${BUILDKITE_JOB_ID}"
  }
  annotate_kernel_wheel "selected; image not yet published"
  cp requirements/xpu.txt "$kernel_dir/xpu.txt"
  if [[ "$(grep -Ec '^vllm[-_]xpu[-_]kernels==' requirements/xpu.txt)" != "1" ]]; then
    echo "Expected exactly one vllm-xpu-kernels pin in requirements/xpu.txt" >&2
    exit 1
  fi
  sed -i -E \
    "s|^vllm[-_]xpu[-_]kernels==.*$|vllm-xpu-kernels @ ${wheel_url}#sha256=${wheel_sha256}|" \
    requirements/xpu.txt
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

# skip build if image already exists
if [[ "$kernel_nightly" == "0" ]] && docker manifest inspect "$IMAGE" &> /dev/null; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  # build
  docker build \
    --file docker/Dockerfile.xpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --tag "$IMAGE" \
    --progress plain .

  if [[ "$kernel_nightly" == "1" ]]; then
    docker run --rm --env "EXPECTED_KERNEL_VERSION=${kernel_version}" \
      --entrypoint /bin/bash "$IMAGE" -c '
        set -eo pipefail
        uv pip show vllm-xpu-kernels | grep -Fx "Version: $EXPECTED_KERNEL_VERSION"
        /opt/venv/bin/python -c "import torch; import vllm_xpu_kernels._C"
      '
  fi

  # push
  docker push "$IMAGE"
fi

if [[ "$kernel_nightly" == "1" ]]; then
  annotate_kernel_wheel "image verified and published"
  buildkite-agent meta-data set xpu-kernel-nightly-image "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
