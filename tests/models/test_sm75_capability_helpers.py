from vllm.models.deepseek_v4.turing.is_sm75 import is_turing_target
from vllm.platforms import current_platform


def test_is_turing_target_cuda_cap7():
    # On a Turing host the platform capability is 7.5.
    cap = current_platform.get_device_capability()
    if current_platform.is_cuda():
        assert is_turing_target(cap) == (cap is not None and cap.major == 7)
    else:
        assert is_turing_target(cap) is False
