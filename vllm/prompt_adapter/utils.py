# code borrowed from: https://github.com/huggingface/peft/blob/v0.12.0/src/peft/utils/save_and_load.py#L420

import os
from typing import Optional

import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file

from vllm.platforms import current_platform

WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"


# Get current device name based on available devices
def infer_device() -> str:
    if current_platform.is_cuda_alike():
        return "cuda"
    return "cpu"


def load_peft_weights(model_id: str,
                      device: Optional[str] = None,
                      **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to
            load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when 
            loading from the HuggingFace Hub.
    """
    path = (os.path.join(model_id, hf_hub_download_kwargs["subfolder"]) if
            hf_hub_download_kwargs.get("subfolder") is not None else model_id)

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token")
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token")

        hub_filename = (os.path.join(hf_hub_download_kwargs["subfolder"],
                                     SAFETENSORS_WEIGHTS_NAME)
                        if hf_hub_download_kwargs.get("subfolder") is not None
                        else SAFETENSORS_WEIGHTS_NAME)
        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=hub_filename,
            revision=hf_hub_download_kwargs.get("revision"),
            repo_type=hf_hub_download_kwargs.get("repo_type"),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME,
                                           **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(  # noqa: B904
                    f"Can't find weights for {model_id} in {model_id} or \
                    in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or \
                    {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}.")

    if use_safetensors:
        adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename,
                                      map_location=torch.device(device),
                                      weights_only=True)

    return adapters_weights
