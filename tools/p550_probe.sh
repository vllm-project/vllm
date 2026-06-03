#!/usr/bin/env bash
set -uo pipefail

section() {
    printf '\n## %s\n' "$1"
}

run() {
    printf '$ %s\n' "$*"
    "$@" 2>&1 || true
}

first_cmd() {
    command -v "$1" 2>/dev/null || true
}

section "identity"
run whoami
run hostname
run date

section "kernel"
run uname -a

section "os-release"
if [ -r /etc/os-release ]; then
    cat /etc/os-release
else
    printf '/etc/os-release is not readable\n'
fi

section "cpu"
run lscpu

section "cpuinfo isa"
if [ -r /proc/cpuinfo ]; then
    grep -Ei '^(isa|uarch|model name|processor|hart|vendor_id|mmu|cpu isa)' /proc/cpuinfo | head -n 80 || true
else
    printf '/proc/cpuinfo is not readable\n'
fi

cpuinfo="$(cat /proc/cpuinfo 2>/dev/null || true)"
rvv_vlen=0
for n in 1024 512 256 128; do
    if printf '%s' "$cpuinfo" | grep -qi "zvl${n}b"; then
        rvv_vlen="$n"
        break
    fi
done

section "risc-v vector recommendation"
if [ "$rvv_vlen" = "128" ] || [ "$rvv_vlen" = "256" ]; then
    printf 'Detected supported RVV VLEN: %s\n' "$rvv_vlen"
    printf 'Recommended build setting: leave VLLM_RVV_VLEN unset, or export VLLM_RVV_VLEN=%s\n' "$rvv_vlen"
else
    if [ "$rvv_vlen" = "0" ]; then
        printf 'No zvl128b/zvl256b ISA string detected.\n'
    else
        printf 'Detected unsupported RVV VLEN: %s\n' "$rvv_vlen"
    fi
    printf 'Recommended build setting: export VLLM_RVV_VLEN=0\n'
fi
for ext in zvfh zvfhmin zvfbfmin zfbfmin; do
    if printf '%s' "$cpuinfo" | grep -qi "$ext"; then
        printf 'Detected extension: %s\n' "$ext"
    fi
done

section "memory and disk"
run free -h
run nproc
run df -h . /tmp

section "toolchain"
for tool in python3 python pip3 pip gcc g++ cmake ninja make git curl; do
    path="$(first_cmd "$tool")"
    if [ -n "$path" ]; then
        printf '%s: %s\n' "$tool" "$path"
        "$tool" --version 2>&1 | head -n 1 || true
    else
        printf '%s: missing\n' "$tool"
    fi
done

section "python packages"
python_bin="$(first_cmd python3)"
if [ -z "$python_bin" ]; then
    python_bin="$(first_cmd python)"
fi
if [ -n "$python_bin" ]; then
    "$python_bin" - <<'PY'
import importlib.util
import platform
import sys

print("python", sys.version)
print("machine", platform.machine())
print("platform", platform.platform())

for name in ("torch", "numpy", "transformers", "vllm"):
    spec = importlib.util.find_spec(name)
    print(name, "found" if spec else "missing")
    if spec is None:
        continue
    try:
        mod = __import__(name)
        print(" ", getattr(mod, "__version__", "no __version__"))
        if name == "torch":
            print(" torch cuda:", getattr(mod.version, "cuda", None))
            print(" torch hip:", getattr(mod.version, "hip", None))
            print(" torch threads:", mod.get_num_threads())
            print(" torch config:")
            print(mod.__config__.show())
    except Exception as exc:
        print(" import-error", type(exc).__name__, exc)
PY
else
    printf 'python is missing\n'
fi

section "libraries"
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -Ei 'openblas|blas|lapack|tcmalloc|numa|gomp|iomp|omp|dnnl|mkl' || true
else
    printf 'ldconfig is missing\n'
fi

section "debian packages"
if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W -f='${Package} ${Version}\n' \
        gcc g++ cmake ninja-build libnuma-dev libtcmalloc-minimal4 \
        python3-dev python3-venv git curl 2>/dev/null || true
else
    printf 'dpkg-query is missing\n'
fi

section "next build commands"
printf 'export VLLM_TARGET_DEVICE=cpu\n'
if [ "$rvv_vlen" = "128" ] || [ "$rvv_vlen" = "256" ]; then
    printf '# export VLLM_RVV_VLEN=%s\n' "$rvv_vlen"
else
    printf 'export VLLM_RVV_VLEN=0\n'
fi
