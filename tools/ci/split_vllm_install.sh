#!/usr/bin/env bash
# Copy the installed vllm distribution out of site-packages into three trees
# that mirror the filesystem root, so the CI test image (docker/Dockerfile,
# test stage) can add each one as its own layer:
#   <out>/native  compiled C++/CUDA extensions; change only with csrc/
#   <out>/rust    Rust frontend, relinked with every commit's version
#   <out>/python  everything else: Python sources, metadata, console scripts
# Hosts that already have a commit's image then pull only the trees that
# differ in the next one, instead of the whole wheel.
#
# Usage: split_vllm_install.sh <out_dir>
set -euo pipefail

out=$1
site=$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
cd "${site}"

dist_infos=(vllm-*.dist-info)
if [ ! -d vllm ] || [ ! -d "${dist_infos[0]}" ]; then
    echo "vllm distribution not found in ${site}" >&2
    exit 1
fi

{
    find vllm vllm-*.dist-info \( -type f -o -type l \)
    # Console scripts are recorded relative to site-packages, e.g. ../../../bin/vllm.
    grep -o '^\.\./[^,]*' vllm-*.dist-info/RECORD || true
} | while IFS= read -r path; do
    case "${path}" in
        vllm/vllm-rs | vllm/_rust_*.so) group=rust ;;
        *.so) group=native ;;
        *) group=python ;;
    esac
    # -s (--no-symlinks) normalizes ../ segments without dereferencing symlinks.
    dest="${out}/${group}$(realpath -ms "${site}/${path}")"
    mkdir -p "$(dirname "${dest}")"
    cp -a "${site}/${path}" "${dest}"
done
mkdir -p "${out}/native" "${out}/rust" "${out}/python"
if [ -z "$(find "${out}/python" -mindepth 1 -print -quit)" ]; then
    echo "split_vllm_install.sh produced an empty python tree" >&2
    exit 1
fi
