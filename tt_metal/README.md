
## vLLM and tt-metal Branches
Git-checkout the following branches in each repo separately:
- vLLM branch: [dev](https://github.com/tenstorrent/vllm/tree/dev)
- tt-metal branch: [vllm_dev](https://github.com/tenstorrent/tt-metal/tree/vllm_dev)

## Environment Creation

To setup the tt-metal environment with vLLM, follow the instructions in `setup-metal.sh`

## Accessing the Meta-Llama-3.1 Hugging Face Model

To run Meta-Llama-3.1, it is required to have access to the model on Hugging Face. 
Steps:
1. Request access on [https://huggingface.co/meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B).
2. Once you have received access, create and copy your access token from the settings tab on Hugging Face.
3. Run this code in python and paste your access token:
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```

## Importing the tt-metal models

Create a symbolic link to the tt-metal models folder inside vLLM:
```sh
cd tt_metal
ln -s <path/to/tt-metal>/models ./models
```

## Running the offline inference example
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py
```