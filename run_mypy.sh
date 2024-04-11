mypy vllm/attention/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/core/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/distributed/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/entrypoints/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/executor/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/usage/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/*.py --follow-imports=skip --config-file pyproject.toml
mypy vllm/transformers_utils/*.py --follow-imports=skip --config-file pyproject.toml

# mypy vllm/engine/*.py --follow-imports=skip --config-file pyproject.toml
# mypy vllm/worker/*.py --follow-imports=skip --config-file pyproject.toml
# mypy vllm/spec_decoding/*.py --follow-imports=skip --config-file pyproject.toml
# mypy vllm/model_executor/*.py --follow-imports=skip --config-file pyproject.toml
# mypy vllm/lora/*.py --follow-imports=skip --config-file pyproject.toml