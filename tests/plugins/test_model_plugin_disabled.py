import pytest

from vllm import LLM

# The test in this file should be run with env VLLM_PLUGINS='', for example:
# VLLM_PLUGINS='' pytest -v -s test_model_plugin_disabled.py


def test_plugin_disabled(dummy_opt_path):
    with pytest.raises(Exception) as excinfo:
        LLM(model=dummy_opt_path, load_format="dummy")
    assert "are not supported for now" in str(excinfo.value)
