
## vLLM and tt-metal Branches
Git-checkout the following branches in each repo separately:
- vLLM branch: [dev](https://github.com/tenstorrent/vllm/tree/dev) (last verified commit: [3f7beb2](https://github.com/tenstorrent/vllm/tree/3f7beb23cbaf3be2e104061905da5f91644e5a68))
- tt-metal branch: [main](https://github.com/tenstorrent/tt-metal) (last verified commit: [0107040](https://github.com/tenstorrent/tt-metal/tree/01070409e582616d8962f371e8497abbf252bb81))

## Environment Creation

**To create the vLLM+tt-metal environment (first time):**
1. Set tt-metal environment variables (see INSTALLING.md in tt-metal repo)
2. From the main vLLM directory, run:
    ```sh
    export vllm_dir=$(pwd)
    source $vllm_dir/tt_metal/setup-metal.sh
    ```
3. From the main tt-metal directory, build and create the environment as usual:
    ```sh
    ./build_metal.sh && ./create_venv.sh
    source $PYTHON_ENV_DIR/bin/activate
    ```
4. Install vLLM:
    ```sh
    pip3 install --upgrade pip
    cd $vllm_dir && pip install -e .
    ```

**To activate the vLLM+tt-metal environment (after the first time):**
1. Ensure `$vllm_dir` contains the path to vLLM and run:
    ```sh
    source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
    ```

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

## Preparing the tt-metal models

1. Create a symbolic link to the tt-metal models folder inside vLLM:
    ```sh
    cd tt_metal
    ln -s <path/to/tt-metal>/models ./models
    ```
2. For the desired model, follow the setup instructions (if any) for the corresponding tt-metal demo. E.g. For Llama-3.1-70B, follow the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/llama3_70b) for preparing the weights and environment variables.

## Running the offline inference example
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py
```

## Running the server example (experimental)
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/server_example_tt.py
```
