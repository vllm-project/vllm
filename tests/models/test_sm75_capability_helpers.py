from vllm.models.deepseek_v4.turing.is_sm75 import is_turing_target
from vllm.platforms import current_platform


def test_is_turing_target_cuda_cap7():
    # On a Turing host the platform capability is 7.5.
    cap = current_platform.get_device_capability()
    if current_platform.is_cuda():
        assert is_turing_target(cap) == (cap is not None and cap.major == 7)
    else:
        assert is_turing_target(cap) is False


def test_is_turing_target_explicit_inputs():
    from vllm.models.deepseek_v4.turing.is_sm75 import is_turing_target

    class FakeCap:
        def __init__(self, major):
            self.major = major

    assert is_turing_target(None) is False
    assert is_turing_target(FakeCap(8)) is False
    assert is_turing_target(FakeCap(9)) is False
    assert is_turing_target(FakeCap(7)) is True


def test_turing_backend_imports_resolve():
    from vllm.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        DeepSeekV4MTP,
        DSparkDeepseekV4ForCausalLM,
    )

    assert DeepseekV4ForCausalLM.__name__ == "DeepseekV4ForCausalLM"
    assert DeepSeekV4MTP.__name__ == "DeepSeekV4MTP"
    assert DSparkDeepseekV4ForCausalLM.__name__ == "DSparkDeepseekV4ForCausalLM"
