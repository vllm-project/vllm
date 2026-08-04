# Comprehensive Tests

An end-to-end integration suite for LMCache & vLLM latest branch.

## Layout

- **Scripts**: `scripts/vllm-integration-tests.sh`
- **Configs**: `configs/`
- **Pipeline**: `pipelines/comprehensive-tests.yml`

## Prepare

1. Add your YAMLs to `configs/` (e.g. `local_cpu.yaml`, `local_disk.yaml`).

2. Add the filenames to `cases/comprehensive-cases.txt`.
