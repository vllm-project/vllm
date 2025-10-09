#!/usr/bin/env bash
# Cross-platform vLLM dev container helper (Podman-first, Linux / WSL)
set -euo pipefail
shopt -s extglob

IMAGE_TAG="vllm-dev:latest"
CONTAINER_NAME="vllm-dev"
SOURCE_DIR="$(pwd)"

BUILD=0
BUILD_NO_CACHE=0
BUILD_PULL=0
GPU_CHECK=0
SETUP=0
INTERACTIVE=0
CMD=""
MIRROR=0
PROGRESS=0
RECREATE=0
WORK_VOLUME=""
WORK_DIR_HOST=""
EXTRA_ENVS=()

show_help() {
	cat <<'EOF'
Usage: extras/podman/run.sh [options]

Options:
  -b, --build               Build (or rebuild) the image first
      --no-cache            Build without using cache
      --pull                Always attempt to pull newer base image
  -c, --command CMD         Run CMD inside the container then exit
  -g, --gpu-check           Run CUDA / Torch diagnostics inside container
  -s, --setup               Run project setup inside the container
  -m, --mirror              Enable LOCAL_MIRROR=1 during setup
  -p, --progress            Show progress bars during setup
      --recreate            Remove any existing container before running
      --work-volume NAME    Mount named volume NAME at /opt/work
      --work-dir-host PATH  Bind mount host PATH at /opt/work
      --env KEY=VALUE       Inject additional environment variable (repeatable)
  -n, --name NAME           Override container name (default: vllm-dev)
  -h, --help                Show this message and exit

Interactive shell is the default when no other action is requested.
Examples:
  extras/podman/run.sh --build --pull
  extras/podman/run.sh --setup --progress
  extras/podman/run.sh --command "python -m pytest tests/..."
  extras/podman/run.sh --gpu-check
EOF
}

normalize_shell_newlines() {
	local root="$1"
	command -v find >/dev/null 2>&1 || return 0
	while IFS= read -r -d '' file; do
		if grep -q $'\r' "$file"; then
			local tmp
			tmp=$(mktemp)
			tr -d '\r' <"$file" >"$tmp"
			touch -r "$file" "$tmp" 2>/dev/null || true
			mv "$tmp" "$file"
		fi
	done < <(find "$root" -type f -name '*.sh' -print0 2>/dev/null)
}

docker_arg_default() {
	local name="$1" fallback="$2" dockerfile="extras/Dockerfile"
	[[ -f "$dockerfile" ]] || { printf '%s\n' "$fallback"; return; }
	local line value
	line=$(grep -E "^\s*ARG\s+${name}\s*=" "$dockerfile" | head -n1 || true)
	if [[ -n "$line" ]]; then
		value="${line#*=}"
		value="${value##+([[:space:]])}"
		value="${value%%+([[:space:]])}"
		printf '%s\n' "$value"
	else
		printf '%s\n' "$fallback"
	fi
}

derive_torch_index() {
	local cuda_ver="$1"
	if [[ "$cuda_ver" =~ ^13\. ]]; then
		printf 'cu130\n'
	elif [[ "$cuda_ver" =~ ^12\.9 ]]; then
		printf 'cu129\n'
	else
		local parts
		IFS='.' read -r -a parts <<<"$cuda_ver"
		if [[ ${#parts[@]} -ge 2 ]]; then
			printf 'cu%s%s0\n' "${parts[0]}" "${parts[1]}"
		else
			printf 'cu129\n'
		fi
	fi
}

ensure_podman() {
	if ! command -v podman >/dev/null 2>&1; then
		echo '❌ podman not found in PATH' >&2
		exit 1
	fi
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		-b|--build) BUILD=1; shift ;;
		--no-cache) BUILD_NO_CACHE=1; shift ;;
		--pull) BUILD_PULL=1; shift ;;
		-c|--command) CMD="${2:-}"; INTERACTIVE=0; shift 2 ;;
		-g|--gpu-check) GPU_CHECK=1; shift ;;
		-s|--setup) SETUP=1; shift ;;
		-m|--mirror) MIRROR=1; shift ;;
		-p|--progress) PROGRESS=1; shift ;;
		--recreate) RECREATE=1; shift ;;
		--work-volume) WORK_VOLUME="${2:-}"; shift 2 ;;
		--work-dir-host) WORK_DIR_HOST="${2:-}"; shift 2 ;;
		--env) EXTRA_ENVS+=("${2:-}"); shift 2 ;;
		-n|--name) CONTAINER_NAME="${2:-}"; shift 2 ;;
		-h|--help) show_help; exit 0 ;;
		*) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
		esac
	done

	if [[ $GPU_CHECK -eq 0 && $SETUP -eq 0 && -z "$CMD" ]]; then
		INTERACTIVE=1
	fi
}

load_build_config() {
	local cfg="extras/configs/build.env"
	if [[ -f "$cfg" ]]; then
		set +u
		# shellcheck disable=SC1091
		source "$cfg"
		set -u
	fi

	CUDA_VERSION="${CUDA_VERSION:-$(docker_arg_default CUDA_VERSION 13.0.0)}"
	BASE_FLAVOR="${BASE_FLAVOR:-$(docker_arg_default BASE_FLAVOR rockylinux9)}"
	TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(docker_arg_default TORCH_CUDA_ARCH_LIST '8.0 8.6 8.9 9.0 12.0 13.0')}"
	CUDA_ARCHS="${CUDA_ARCHS:-$(docker_arg_default CUDA_ARCHS '80;86;89;90;120')}"
	REQUIRE_FFMPEG="${REQUIRE_FFMPEG:-$(docker_arg_default REQUIRE_FFMPEG 1)}"
	INSTALL_CUDA_OPTIONAL_DEVEL="${INSTALL_CUDA_OPTIONAL_DEVEL:-$(docker_arg_default INSTALL_CUDA_OPTIONAL_DEVEL 1)}"
	CUDNN_FLAVOR="${CUDNN_FLAVOR:-$(docker_arg_default CUDNN_FLAVOR 9)}"
	TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-$(derive_torch_index "$CUDA_VERSION")}"
}

build_image_if_requested() {
	[[ $BUILD -eq 1 ]] || return 0
	echo "🔨 Building image (honoring extras/configs/build.env)..."
	load_build_config
	local args=(build -f extras/Dockerfile -t "$IMAGE_TAG"
		--build-arg "CUDA_VERSION=$CUDA_VERSION"
		--build-arg "BASE_FLAVOR=$BASE_FLAVOR"
		--build-arg "TORCH_CUDA_INDEX=$TORCH_CUDA_INDEX"
		--build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
		--build-arg "CUDA_ARCHS=$CUDA_ARCHS"
		--build-arg "INSTALL_CUDA_OPTIONAL_DEVEL=$INSTALL_CUDA_OPTIONAL_DEVEL"
		--build-arg "CUDNN_FLAVOR=$CUDNN_FLAVOR"
		--build-arg "REQUIRE_FFMPEG=$REQUIRE_FFMPEG")
	[[ $BUILD_NO_CACHE -eq 1 ]] && args=(build --no-cache "${args[@]:1}")
	[[ $BUILD_PULL -eq 1 ]] && args=(build --pull=always "${args[@]:1}")
	args+=("$SOURCE_DIR")
	if ! podman "${args[@]}"; then
		echo "❌ Build failed" >&2
		exit 1
	fi
	echo "✅ Build ok"
}

remove_running_container_if_needed() {
	local running
	running=$(podman ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' 2>/dev/null || true)
	if [[ $RECREATE -eq 1 && "$running" == "$CONTAINER_NAME" ]]; then
		echo "♻️  Removing existing container '$CONTAINER_NAME'"
		podman rm -f "$CONTAINER_NAME" >/dev/null
	fi
}

handle_existing_container() {
	local running
	running=$(podman ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' 2>/dev/null || true)
	[[ "$running" == "$CONTAINER_NAME" ]] || return 1

	if [[ $GPU_CHECK -eq 1 ]]; then
		echo "🔍 GPU check (existing container)"
		local script
		read -r -d '' script <<'EOF'
export NVIDIA_VISIBLE_DEVICES=all
source /home/vllmuser/venv/bin/activate 2>/dev/null || true
which nvidia-smi && nvidia-smi || echo 'nvidia-smi unavailable'
python - <<'PY'
import torch, os
print('PyTorch:', getattr(torch, '__version__', 'n/a'))
print('CUDA available:', torch.cuda.is_available())
print('Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
	try:
		print('GPU 0:', torch.cuda.get_device_name(0))
	except Exception as exc:
		print('GPU name error:', exc)
PY
EOF
		exec podman exec "$CONTAINER_NAME" bash -lc "$script"
	fi

	if [[ $SETUP -eq 1 ]]; then
		echo "🔧 Running dev setup in existing container"
		local env_prefix=""
		[[ $MIRROR -eq 1 ]] && env_prefix+='export LOCAL_MIRROR=1; '
		[[ $PROGRESS -eq 1 ]] && env_prefix+='export PROGRESS_WATCH=1; '
		exec podman exec "$CONTAINER_NAME" bash -lc "${env_prefix}chmod +x ./extras/dev-setup.sh 2>/dev/null || true; ./extras/dev-setup.sh"
	fi

	if [[ -n "$CMD" ]]; then
		echo "🚀 Running command in existing container"
		podman exec "$CONTAINER_NAME" bash -lc "source /home/vllmuser/venv/bin/activate 2>/dev/null || true; $CMD"
		exit $?
	fi

	read -r -p "Attach to running container '$CONTAINER_NAME'? [Y/n] " reply || reply=""
	if [[ -z "$reply" || "$reply" =~ ^[Yy]$ ]]; then
		exec podman exec -it "$CONTAINER_NAME" bash
	else
		exit 0
	fi
}

ensure_image_exists() {
	if [[ $BUILD -eq 0 ]]; then
		if ! podman image exists "$IMAGE_TAG"; then
			echo "❌ Image missing. Use --build." >&2
			exit 1
		fi
	fi
}

collect_fa3_envs() {
	while IFS='=' read -r key value; do
		if [[ $key == FA3_* ]]; then
			RUN_ARGS+=(--env "$key=$value")
		fi
	done < <(env)
}

validate_env_kv() {
	local kv="$1"
	if [[ ! "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
		echo "⚠️  Ignoring invalid --env entry: $kv" >&2
		return 1
	fi
	return 0
}

prepare_run_args() {
	RUN_ARGS=(run --rm --security-opt=label=disable --device=nvidia.com/gpu=all --shm-size 8g)
	RUN_ARGS+=(--name "$CONTAINER_NAME" -v "${SOURCE_DIR}:/workspace:Z" -w /workspace --user vllmuser --env ENGINE=podman)
	RUN_ARGS+=(--entrypoint /workspace/extras/podman/entrypoint/apply-patches-then-exec.sh)

	if [[ -n "$WORK_VOLUME" ]]; then
		RUN_ARGS+=(-v "${WORK_VOLUME}:/opt/work:Z")
	elif [[ -n "$WORK_DIR_HOST" ]]; then
		if [[ -d "$WORK_DIR_HOST" ]]; then
			RUN_ARGS+=(-v "${WORK_DIR_HOST}:/opt/work:Z")
		else
			echo "⚠️  --work-dir-host '${WORK_DIR_HOST}' not found; skipping" >&2
		fi
	fi

	local tmpfs_size="${VLLM_TMPFS_TMP_SIZE:-0}"
	if [[ -n "$tmpfs_size" && "$tmpfs_size" != 0 ]]; then
		RUN_ARGS+=(--tmpfs "/tmp:size=${tmpfs_size}")
	fi

	RUN_ARGS+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
				--env "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}" \
				--env "NVIDIA_REQUIRE_CUDA=")

	if [[ -d /usr/lib/wsl ]]; then
		RUN_ARGS+=(--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro)
	fi

	for kv in "${EXTRA_ENVS[@]}"; do
		if validate_env_kv "$kv"; then
			RUN_ARGS+=(--env "$kv")
		fi
	done

	collect_fa3_envs
}

gpu_check_script() {
	cat <<'EOF'
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:$LD_LIBRARY_PATH
echo '=== GPU Check ==='
which nvidia-smi && nvidia-smi || echo 'nvidia-smi unavailable'
echo '--- /dev/nvidia* ---'
ls -l /dev/nvidia* 2>/dev/null || echo 'no /dev/nvidia* nodes'
echo '--- Environment (NVIDIA_*) ---'
env | grep -E '^NVIDIA_' || echo 'no NVIDIA_* env vars'
if [ "$NVIDIA_VISIBLE_DEVICES" = "void" ]; then echo 'WARN: NVIDIA_VISIBLE_DEVICES=void (no GPU mapped)'; fi
echo '--- LD_LIBRARY_PATH ---'
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
source /home/vllmuser/venv/bin/activate 2>/dev/null || true
python - <<'PY'
import json, torch, os
out = {
	"torch_version": getattr(torch, "__version__", "n/a"),
	"torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", "n/a"),
	"cuda_available": torch.cuda.is_available(),
	"ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
}
try:
	out["device_count"] = torch.cuda.device_count()
except Exception as exc:
	out["device_count_error"] = str(exc)
if out["cuda_available"] and out.get("device_count", 0) > 0:
	try:
		cap = torch.cuda.get_device_capability(0)
		out["device_0"] = {
			"name": torch.cuda.get_device_name(0),
			"capability": f"sm_{cap[0]}{cap[1]}",
		}
	except Exception as exc:
		out["device_0_error"] = str(exc)
else:
	out["diagnostics"] = ["Missing /dev/nvidia* or podman machine without GPU passthrough"]
print(json.dumps(out, indent=2))
PY
EOF
}

setup_command() {
	cat <<'EOF'
TMP_RUN=$(mktemp /tmp/run-dev-setup.XXXX.sh)
tr -d '\r' < ./extras/podman/dev-setup.sh > "$TMP_RUN" || cp ./extras/podman/dev-setup.sh "$TMP_RUN"
chmod +x "$TMP_RUN" 2>/dev/null || true
bash extras/patches/apply_patches.sh || true
export TMPDIR=/opt/work/tmp
export TMP=/opt/work/tmp
export TEMP=/opt/work/tmp
mkdir -p /opt/work/tmp
"$TMP_RUN"
EOF
}

dispatch() {
	if [[ $GPU_CHECK -eq 1 ]]; then
		RUN_ARGS+=("$IMAGE_TAG" bash -lc "$(gpu_check_script)")
	elif [[ $SETUP -eq 1 ]]; then
		raise_setup_env
	else
		run_default_or_command
	fi
}

raise_setup_env() {
	load_build_config
	[[ -n "$TORCH_CUDA_ARCH_LIST" ]] && RUN_ARGS+=(--env "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST")
	[[ -n "$CUDA_ARCHS" ]] && RUN_ARGS+=(--env "CUDAARCHS=$CUDA_ARCHS")
	[[ $MIRROR -eq 1 ]] && RUN_ARGS+=(--env LOCAL_MIRROR=1)
	[[ $PROGRESS -eq 1 ]] && RUN_ARGS+=(--env PROGRESS_WATCH=1)
	if [[ $PROGRESS -eq 1 ]]; then
		RUN_ARGS+=(-it "$IMAGE_TAG" bash -lc "$(setup_command)")
	else
		RUN_ARGS+=("$IMAGE_TAG" bash -lc "$(setup_command)")
	fi
}

run_default_or_command() {
	if [[ -n "$CMD" ]]; then
		RUN_ARGS+=("$IMAGE_TAG" bash -lc "source /home/vllmuser/venv/bin/activate 2>/dev/null || true; $CMD")
	else
		RUN_ARGS+=(-it "$IMAGE_TAG" bash)
		echo "🚀 Interactive shell tips once inside:"
		echo "  ./extras/dev-setup.sh"
		echo "  python -c 'import torch; print(torch.cuda.is_available())'"
		echo "  python -c 'import vllm'"
	fi
}

main() {
	parse_args "$@"
	ensure_podman
	normalize_shell_newlines "extras"
	build_image_if_requested
	remove_running_container_if_needed
	if handle_existing_container; then
		return 0
	fi
	ensure_image_exists
	prepare_run_args
	dispatch
	echo "Command: podman ${RUN_ARGS[*]}"
	exec podman "${RUN_ARGS[@]}"
}

main "$@"
