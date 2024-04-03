import  huggingface_hub.constants
from vllm.model_executor.weight_utils import enable_hf_transfer

def test_hf_transfer_auto_activation():
    enable_hf_transfer()
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa
        HF_TRANFER_ACTIVE = True
    except ImportError:
        HF_TRANFER_ACTIVE = False
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER == HF_TRANFER_ACTIVE

if __name__ == "__main__":
    test_hf_transfer_auto_activation()