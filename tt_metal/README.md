
## vLLM and tt-metal Branches
Git-checkout the following branches in each repo separately:
- vLLM branch: [dev](https://github.com/tenstorrent/vllm/tree/dev) (last verified commit: [9531611](https://github.com/tenstorrent/vllm/tree/953161188c50f10da95a88ab305e23977ebd3750))
- tt-metal branch: [main](https://github.com/tenstorrent/tt-metal) (last verified commit: [3ef6837](https://github.com/tenstorrent/tt-metal/tree/3ef683762eaa4bd602ec6f3f33aec875775265c5))

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

## Accessing the Meta-Llama Hugging Face Models

To run Meta-Llama-3.1/3.2, it is required to have access to the model on Hugging Face. To gain access:
1. Request access on Hugging Face:
    - Llama-3.1: [https://huggingface.co/meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)
    - Llama-3.2: [https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
2. Once you have received access, create and copy your access token from the settings tab on Hugging Face.
3. Run this code in python and paste your access token:
    ```python
    from huggingface_hub import login
    login()
    ```

## Preparing the tt-metal models

1. Ensure that `$PYTHONPATH` contains the path to tt-metal (should already have been done when installing tt-metal)
2. For the desired model, follow the setup instructions (if any) for the corresponding tt-metal demo. E.g. For Llama-3.1-70B, follow the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/llama3_70b) for preparing the weights and environment variables.

## Running the offline inference example

### Llama-3.1-70B

To generate tokens for sample prompts:
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py
```

To measure performance for a single batch (with the default prompt length of 128 tokens):
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --measure_perf
```

### Llama-3.2-11B-Vision-Instruct

To generate tokens for sample prompts:
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --multi_modal --max_seqs_in_batch 16 --num_repeat_prompts 8
```

To measure performance for a single batch (with the default prompt length of 128 tokens):
```python
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --measure_perf --multi_modal --max_seqs_in_batch 16 --num_repeat_prompts 4
```

**Note**: By default, the multi-modal inference example will run with `MESH_DEVICE=N300`. To run on T3000, set `MESH_DEVICE=T3K_LINE` and `--max_seqs_in_batch 32 --num_repeat_prompts 16`.

## Running the server example

```python
VLLM_RPC_TIMEOUT=100000 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/server_example_tt.py
```

**Note**: By default, the server will run with Llama-3.1-70B. To run with Llama-3.2-11B-Vision-Instruct instead, add `--multi_modal`.
