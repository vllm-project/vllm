#!/bin/bash
# Same build as the x86 CPU image, including its registry layer cache.
CPU_ARCH=arm64 exec "$(dirname "$0")/image_build_cpu.sh" "$@"
