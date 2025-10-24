import importlib
import os

def _reload_vllm_with_env(val: str | None):
    if val is None:
        os.environ.pop("VLLM_USE_V1", None)
    else:
        os.environ["VLLM_USE_V1"] = val

    import vllm.envs as envs
    import vllm
    importlib.reload(envs)
    importlib.reload(vllm)
    return vllm

def test_samplingparams_importable_when_v1_zero():
    _reload_vllm_with_env("0")
    from vllm import SamplingParams  # must not raise
    assert SamplingParams is not None

def test_samplingparams_importable_when_v1_unset_and_one():
    _reload_vllm_with_env(None)
    from vllm import SamplingParams  # must not raise
    _reload_vllm_with_env("1")
    from vllm import SamplingParams  # must not raise