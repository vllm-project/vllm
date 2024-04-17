import os

import huggingface_hub.constants
import pytest

from vllm.model_executor.model_loader.weight_utils import enable_hf_transfer


def test_hf_transfer_auto_activation():
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
        # in case it is already set, we can't test the auto activation
        pytest.skip(
            "HF_HUB_ENABLE_HF_TRANSFER is set, can't test auto activation")
    enable_hf_transfer()
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa
        HF_TRANFER_ACTIVE = True
    except ImportError:
        HF_TRANFER_ACTIVE = False
    assert (huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER ==
            HF_TRANFER_ACTIVE)


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
