#!/bin/sh

if [ -z "${_VLLM_INTERNAL_NUMACTL_ARGS:-}" ]; then
    echo "_VLLM_INTERNAL_NUMACTL_ARGS is not set" >&2
    exit 1
fi

if [ -z "${_VLLM_INTERNAL_NUMACTL_PYTHON_EXECUTABLE:-}" ]; then
    echo "_VLLM_INTERNAL_NUMACTL_PYTHON_EXECUTABLE is not set" >&2
    exit 1
fi

exec numactl ${_VLLM_INTERNAL_NUMACTL_ARGS} "${_VLLM_INTERNAL_NUMACTL_PYTHON_EXECUTABLE}" "$@"
