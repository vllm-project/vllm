import pytest

from vllm import envs
from vllm.engine.arg_utils import EngineArgs

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


def test_v1_defaults():
    engine_args = EngineArgs(model="facebook/opt-125m")

    # Assert V1 defaults
    assert engine_args.enable_prefix_caching
    assert engine_args.max_num_seqs == 1024
    assert engine_args.max_num_batched_tokens is None
