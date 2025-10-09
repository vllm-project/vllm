#!/usr/bin/env bash
# Primary vLLM dev container helper (Podman-first, Linux / WSL)
set -euo pipefail
shopt -s extglob

: "${IMAGE_TAG:=vllm-dev:latest}"
CONTAINER_NAME="vllm-dev"
SOURCE_DIR="${SOURCE_DIR:-$(pwd)}"

HOST_UNAME=$(uname -s 2>/dev/null || echo unknown)
IS_MSYS=0
case "$HOST_UNAME" in
	MINGW*|MSYS*|CYGWIN*) IS_MSYS=1 ;;
esac

convert_host_path() {
	local input="$1"
	if [[ -z "$input" ]]; then
		printf '%s' "$input"
		return
	fi
	if [[ $IS_MSYS -eq 1 ]] && command -v cygpath >/dev/null 2>&1; then
		cygpath -w "$input"
	else
		printf '%s' "$input"
	fi
}

SOURCE_DIR_NATIVE="$(convert_host_path "$SOURCE_DIR")"

if [[ $IS_MSYS -eq 1 ]]; then
	export MSYS2_ARG_CONV_EXCL='*'
fi

# Action flags
BUILD=0
NO_CACHE=0
PULL=0
SETUP=0
GPU_CHECK=0
INTERACTIVE=0
CMD=""
MIRROR=0
PROGRESS=0
RECREATE=0
WORK_VOLUME=""
WORK_DIR_HOST=""
ENV_ENTRIES=()
SECRET_ENV_FILES=()

# Build/config placeholders (populated lazily)
CUDA_VERSION=""
BASE_FLAVOR=""
TORCH_CUDA_INDEX=""
TORCH_CUDA_ARCH_LIST=""
CUDA_ARCHS=""
INSTALL_CUDA_OPTIONAL_DEVEL=""
CUDNN_FLAVOR=""
REQUIRE_FFMPEG=""

show_help() {
	cat <<'EOF'
Usage: extras/podman/run.sh [options]

Options:
  -b, --build               Build the dev image first
      --no-cache            Build without using cache
      --pull                Always attempt to pull a newer base image
  -c, --command CMD         Run CMD inside the dev container then exit
  -g, --gpu-check           Run CUDA / PyTorch diagnostics inside the container
  -s, --setup               Run project setup helper inside the container
  -m, --mirror              Enable LOCAL_MIRROR=1 during setup
  -p, --progress            Show progress bars during setup
      --recreate            Remove any existing container before running
      --work-volume NAME    Mount named volume NAME at /opt/work
      --work-dir-host PATH  Bind mount PATH at /opt/work
      --env KEY=VALUE       Inject additional environment variables (repeatable)
  -n, --name NAME           Override container name (default: vllm-dev)
  -h, --help                Show this help and exit

Interactive shell is the default when no other action is requested.
EOF
}

err() { echo "${1:-unknown error}" >&2; }

normalize_shell_scripts() {
	local targets=("extras/podman" "extras/patches")
	for root in "${targets[@]}"; do
		[[ -d "$root" ]] || continue
		while IFS= read -r -d '' file; do
			if grep -q $'\r' "$file" 2>/dev/null; then
				local tmp
				tmp=$(mktemp)
				tr -d '\r' <"$file" >"$tmp"
				touch -r "$file" "$tmp" 2>/dev/null || true
				mv "$tmp" "$file"
			fi
		done < <(find "$root" -type f -name '*.sh' -print0 2>/dev/null)
	done
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
	INSTALL_CUDA_OPTIONAL_DEVEL="${INSTALL_CUDA_OPTIONAL_DEVEL:-$(docker_arg_default INSTALL_CUDA_OPTIONAL_DEVEL 1)}"
	CUDNN_FLAVOR="${CUDNN_FLAVOR:-$(docker_arg_default CUDNN_FLAVOR 9)}"
	REQUIRE_FFMPEG="${REQUIRE_FFMPEG:-$(docker_arg_default REQUIRE_FFMPEG 1)}"
	TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-$(derive_torch_index "$CUDA_VERSION")}" 
}

ensure_podman() {
	if ! command -v podman >/dev/null 2>&1; then
		err '❌ Podman not found in PATH'
		exit 1
	fi
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		-b|--build) BUILD=1; shift ;;
		--no-cache) NO_CACHE=1; shift ;;
		--pull) PULL=1; shift ;;
		-c|--command) CMD="${2:-}"; INTERACTIVE=0; shift 2 ;;
		-g|--gpu-check) GPU_CHECK=1; shift ;;
		-s|--setup) SETUP=1; shift ;;
		-m|--mirror) MIRROR=1; shift ;;
		-p|--progress) PROGRESS=1; shift ;;
		--recreate) RECREATE=1; shift ;;
		--work-volume) WORK_VOLUME="${2:-}"; shift 2 ;;
		--work-dir-host) WORK_DIR_HOST="${2:-}"; shift 2 ;;
		--env) ENV_ENTRIES+=("${2:-}"); shift 2 ;;
		-n|--name) CONTAINER_NAME="${2:-}"; shift 2 ;;
		-h|--help) show_help; exit 0 ;;
		*) err "Unknown option: $1"; show_help; exit 1 ;;
		esac
	done

	if [[ $GPU_CHECK -eq 0 && $SETUP -eq 0 && -n "$CMD" ]]; then
		INTERACTIVE=0
	elif [[ $GPU_CHECK -eq 0 && $SETUP -eq 0 && -z "$CMD" ]]; then
		INTERACTIVE=1
	fi
}

build_image_if_requested() {
	[[ $BUILD -eq 1 ]] || return 0
	echo "🔨 Building image (honoring extras/configs/build.env)..."
	load_build_config
	local args=(build -f extras/Dockerfile
		--build-arg "CUDA_VERSION=$CUDA_VERSION"
		--build-arg "BASE_FLAVOR=$BASE_FLAVOR"
		--build-arg "TORCH_CUDA_INDEX=$TORCH_CUDA_INDEX"
		--build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
		--build-arg "CUDA_ARCHS=$CUDA_ARCHS"
		--build-arg "INSTALL_CUDA_OPTIONAL_DEVEL=$INSTALL_CUDA_OPTIONAL_DEVEL"
		--build-arg "CUDNN_FLAVOR=$CUDNN_FLAVOR"
		--build-arg "REQUIRE_FFMPEG=$REQUIRE_FFMPEG"
		-t "$IMAGE_TAG"
		"$SOURCE_DIR_NATIVE")
	if [[ $NO_CACHE -eq 1 ]]; then
		args=(build --no-cache "${args[@]:1}")
	fi
	if [[ $PULL -eq 1 ]]; then
		args=(build --pull=always "${args[@]:1}")
	fi
	if ! podman "${args[@]}"; then
		err '❌ Build failed'
		exit 1
	fi
	echo '✅ Build ok'
}

remove_running_container_if_needed() {
	local running
	running=$(podman ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' 2>/dev/null || true)
	if [[ $RECREATE -eq 1 && "$running" == "$CONTAINER_NAME" ]]; then
		echo "♻️  Removing existing container '$CONTAINER_NAME'"
		podman rm -f "$CONTAINER_NAME" >/dev/null || true
	fi
}

setup_script() {
	cat <<'EOF'
TMP_RUN=$(mktemp /tmp/run-dev-setup.XXXX.sh)
tr -d '\r' < ./extras/podman/dev-setup.sh > "$TMP_RUN" 2>/dev/null || cp ./extras/podman/dev-setup.sh "$TMP_RUN"
chmod +x "$TMP_RUN" 2>/dev/null || true
export PYTHON_PATCH_OVERLAY=1
if [ -x ./extras/patches/apply_patches_overlay.sh ]; then
	bash ./extras/patches/apply_patches_overlay.sh || true
elif [ -x ./extras/patches/apply_patches.sh ]; then
	bash ./extras/patches/apply_patches.sh || true
fi
export TMPDIR=/opt/work/tmp
export TMP=/opt/work/tmp
export TEMP=/opt/work/tmp
mkdir -p /opt/work/tmp
"$TMP_RUN"
EOF
}

gpu_check_script() {
	cat <<'EOF'
export NVIDIA_VISIBLE_DEVICES=all
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

handle_existing_container() {
	local running
	running=$(podman ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' 2>/dev/null || true)
	[[ "$running" == "$CONTAINER_NAME" ]] || return 1

	if [[ $GPU_CHECK -eq 1 ]]; then
		echo '🔍 GPU check (existing container)'
		local script
		script=$(gpu_check_script)
		exec podman exec "$CONTAINER_NAME" bash -lc "$script"
	fi

	if [[ $SETUP -eq 1 ]]; then
		echo '🔧 Running dev setup in existing container'
		local setup_cmd
		setup_cmd=$(command_for_setup)
		exec podman exec "$CONTAINER_NAME" bash -lc "export NVIDIA_VISIBLE_DEVICES=all; export PYTHON_PATCH_OVERLAY=1; $setup_cmd"
	fi

	if [[ -n "$CMD" ]]; then
		echo '🚀 Running command in existing container'
		exec podman exec "$CONTAINER_NAME" bash -lc "export NVIDIA_VISIBLE_DEVICES=all; source /home/vllmuser/venv/bin/activate 2>/dev/null || true; $CMD"
	fi

	read -r -p "Attach to running container '$CONTAINER_NAME'? [Y/n] " reply || reply=""
	if [[ -z "$reply" || "$reply" =~ ^[Yy]$ ]]; then
		exec podman exec -it "$CONTAINER_NAME" bash
	else
		exit 0
	fi
}

ensure_image_exists() {
	if ! podman image exists "$IMAGE_TAG" >/dev/null 2>&1; then
		err '❌ Image missing. Use --build.'
		exit 1
	fi
}

add_env_var() {
	local kv="$1"
	[[ -n "$kv" ]] || return
	if [[ ! "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
		err "⚠️  Ignoring invalid env entry: $kv"
		return
	fi
	local key="${kv%%=*}"
	for existing in "${ENV_VARS[@]-}"; do
		if [[ "${existing%%=*}" == "$key" ]]; then
			return
		fi
	done
	ENV_VARS+=("$kv")
}

collect_fa3_envs() {
	while IFS='=' read -r key value; do
		if [[ $key == FA3_* ]]; then
			add_env_var "$key=$value"
		fi
	done < <(env)
}

collect_secret_env_files() {
	local secrets_dir="$SOURCE_DIR/extras/secrets"
	[[ -d "$secrets_dir" ]] || return
	while IFS= read -r -d '' file; do
		local native_path
		native_path="$(convert_host_path "$file")"
		SECRET_ENV_FILES+=("$native_path")
		local display_name
		display_name="${file##*/}"
		echo "🔐 Detected secrets env file: $display_name" >&2
	done < <(find "$secrets_dir" -maxdepth 1 -type f -name '*.env' ! -name '*.env.example' -print0 2>/dev/null)
}

prepare_run_args() {
	RUN_ARGS=(run --rm --security-opt=label=disable)
	if [[ -z "${VLLM_DISABLE_CDI:-}" ]]; then
		RUN_ARGS+=(--device nvidia.com/gpu=all)
	else
		err '⚠️  Skipping CDI GPU request (VLLM_DISABLE_CDI set)'
	fi
	local source_mount="$SOURCE_DIR_NATIVE"
	RUN_ARGS+=(--shm-size 8g -v "${source_mount}:/workspace:Z" -w /workspace --name "$CONTAINER_NAME" --user vllmuser --entrypoint /workspace/extras/podman/entrypoint/apply-patches-then-exec.sh)

	if [[ -n "$WORK_VOLUME" ]]; then
		RUN_ARGS+=(-v "${WORK_VOLUME}:/opt/work:Z")
	elif [[ -n "$WORK_DIR_HOST" ]]; then
		if [[ -d "$WORK_DIR_HOST" ]]; then
			local work_mount
			work_mount="$(convert_host_path "$WORK_DIR_HOST")"
			RUN_ARGS+=(-v "${work_mount}:/opt/work:Z")
		else
			err "⚠️  --work-dir-host '$WORK_DIR_HOST' not found; skipping"
		fi
	fi

	local tmpfs_size="${VLLM_TMPFS_TMP_SIZE:-}"
	if [[ -n "$tmpfs_size" && "$tmpfs_size" != 0 ]]; then
		RUN_ARGS+=(--tmpfs "/tmp:size=$tmpfs_size")
	fi

	if [[ $IS_MSYS -eq 1 ]]; then
		true
	elif [[ -d /usr/lib/wsl ]]; then
		RUN_ARGS+=(-v /usr/lib/wsl:/usr/lib/wsl:ro)
		if [[ -e /dev/dxg ]]; then
			RUN_ARGS+=(--device /dev/dxg)
		else
			err '⚠️  /dev/dxg not available; GPU passthrough may be disabled'
		fi
	else
		err '⚠️  WSL GPU libraries not detected; continuing without /usr/lib/wsl mount'
	fi

	ENV_VARS=()
	add_env_var 'ENGINE=podman'
	add_env_var "PATCH_OVERLAY_WARN_LIMIT=${PATCH_OVERLAY_WARN_LIMIT:-0}"
	add_env_var "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
	add_env_var "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
	add_env_var "NVIDIA_REQUIRE_CUDA=${NVIDIA_REQUIRE_CUDA:-}"

	if [[ $MIRROR -eq 1 ]]; then
		add_env_var 'LOCAL_MIRROR=1'
	fi
	if [[ $PROGRESS -eq 1 ]]; then
		add_env_var 'PROGRESS_WATCH=1'
	fi

	for kv in "${ENV_ENTRIES[@]}"; do
		add_env_var "$kv"
	done

	collect_fa3_envs
	collect_secret_env_files
}

command_for_gpu_check() {
	local script
	script=$(gpu_check_script)
	printf 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:$LD_LIBRARY_PATH; cat <<"EOF" >/tmp/gpu-check.sh\n%s\nEOF\nbash /tmp/gpu-check.sh\nrm -f /tmp/gpu-check.sh\n' "$script"
}

command_for_setup() {
	local script
	script=$(setup_script)
	printf 'cat <<"EOF" >/tmp/run-setup.sh\n%s\nEOF\nbash /tmp/run-setup.sh\nrm -f /tmp/run-setup.sh\n' "$script"
}

finalize_and_run() {
	for kv in "${ENV_VARS[@]}"; do
		RUN_ARGS+=(--env "$kv")
	done

	for env_file in "${SECRET_ENV_FILES[@]}"; do
		RUN_ARGS+=(--env-file "$env_file")
	done

	if [[ $GPU_CHECK -eq 1 ]]; then
		RUN_ARGS+=(--user root "$IMAGE_TAG" bash -lc "$(command_for_gpu_check)")
	elif [[ $SETUP -eq 1 ]]; then
		load_build_config
		if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
			RUN_ARGS+=(--env "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST")
		fi
		if [[ -n "$CUDA_ARCHS" ]]; then
			RUN_ARGS+=(--env "CUDAARCHS=$CUDA_ARCHS")
		fi
		RUN_ARGS+=("$IMAGE_TAG" bash -lc "$(command_for_setup)")
	elif [[ -n "$CMD" ]]; then
		RUN_ARGS+=("$IMAGE_TAG" bash -lc "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:$LD_LIBRARY_PATH; source /home/vllmuser/venv/bin/activate 2>/dev/null || true; $CMD")
	else
		RUN_ARGS+=(-it "$IMAGE_TAG" bash)
	fi

	echo "Command: podman ${RUN_ARGS[*]}"
	exec podman "${RUN_ARGS[@]}"
}

main() {
	parse_args "$@"
	ensure_podman
	normalize_shell_scripts
	build_image_if_requested
	remove_running_container_if_needed
	if handle_existing_container; then
		return 0
	fi
	ensure_image_exists
	prepare_run_args
	finalize_and_run
}

main "$@"
