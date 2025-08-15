
## vLLM and tt-metal Branches
For the latest versions of vLLM and tt-metal, git-checkout the following branches in each repo separately:
- vLLM branch: [dev](https://github.com/tenstorrent/vllm/tree/dev) (do not commit to this branch)
- tt-metal branch: [main](https://github.com/tenstorrent/tt-metal)

>[!NOTE]
> If testing a specific model, please refer to the [TT-Metal LLMs table](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms) for the appropriate commits to use for tt-metal and vLLM.

## System Requirements
vLLM requires Python 3.9+ (Python 3.10.12 is the default `python3` on Ubuntu 22.04).

## Environment Creation

**To create the vLLM+tt-metal environment (first time):**
1. Install and build tt-metal following the instructions in [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md). Ensure that the necessary environment variables for running tt-metal tests were set.
2. From the main vLLM directory, run:

    ```sh
    export vllm_dir=$(pwd)
    source $vllm_dir/tt_metal/setup-metal.sh
    ```
  
3. (Optional step when installing tt-metal from source) In step 2, `PYTHON_ENV_DIR` is set to `${TT_METAL_HOME}/build/python_env_vllm`. Create the tt-metal virtual environment following the instructions in [INSTALLING.md#option-1-from-source](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md#option-1-from-source). Then, enter that virtual environment with `source $PYTHON_ENV_DIR/bin/activate`.
4. Install vLLM:

    ```sh
    pip3 install --upgrade pip
    cd $vllm_dir && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
    ```

**To activate the vLLM+tt-metal environment (after the first time):**
1. Ensure `$vllm_dir` contains the path to vLLM and run:

    ```sh
    source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
    ```

## Accessing the Meta-Llama Hugging Face Models

To run Meta-Llama-3.1/3.2, it is required to have access to the model on Hugging Face. To gain access:
1. Request access on Hugging Face:
   - Llama-3.1: [https://huggingface.co/meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
   - Llama-3.2: [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
   - Llama-3.2-Vision: [https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
2. Once you have received access, create and copy your access token from the settings tab on Hugging Face.
3. Run this code in python and paste your access token:
  
    ```python
    from huggingface_hub import login
    login()
    ```

## Preparing the tt-metal models

1. Ensure that `$PYTHONPATH` contains the path to tt-metal (should already have been done when installing tt-metal)
2. For the desired model, follow the setup instructions (if any) for the corresponding tt-metal demo. E.g. For Llama-3.1/3.2 and Qwen-2.5, follow the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) for preparing the weights and environment variables, and install any extra requirements (e.g. `pip install -r models/tt_transformers/requirements.txt`).

## Running the offline inference example

### Llama-3.1/3.2 (1B, 3B, 8B, 70B) and Qwen-2.5 (7B, 72B) Text Models

To generate tokens (Llama70B on QuietBox) for sample prompts (with batch size 32):

```sh
MESH_DEVICE=T3K python examples/offline_inference_tt.py
```

To measure performance (Llama70B on QuietBox) for a single batch of 32 prompts (with the default prompt length of 128 tokens):

```sh
MESH_DEVICE=T3K python examples/offline_inference_tt.py --measure_perf
```

**Note 1**: Custom TT options can be set using `--override_tt_config` with a json string, e.g. `--override_tt_config '{"sample_on_device_mode": "all"}'`, however these shouldn't be used unless the model supports them (most currently do not). Supported parameters are:
- `sample_on_device_mode`: ["all", "decode_only"]
- `trace_region_size`: [default: 25000000]
- `worker_l1_size`
- `fabric_config`: ["DISABLED", "FABRIC_1D", "FABRIC_2D", "CUSTOM"]
- `dispatch_core_axis`: ["row", "col"]
- `data_parallel`: [default: 1]

**Note 2 (Llama70B)**: To run Llama70B on Galaxy, set `MESH_DEVICE=TG`.

**Note 3 (Llama70B)**: By default, this will run the newer tt-metal implementation of Llama70B from the [tt_transformers demo](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers). To run other implementations use the `TT_LLAMA_TEXT_VER` environment variable:
- `"llama3_70b_galaxy"` for the Llama TG implementation
- `"llama2_70b"` for the old Llama implementation

**Note 4 (Other Models)**: By default, the inference example will run with Llama-3.1-70B. To run with other Llama models, or Qwen-2.5, ensure that the apprioriate environment variables are set as per the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers), then set `MESH_DEVICE=<device>` (valid options for `<device>` are `N150`, `N300`, `T3K`, or `TG`) and one of the following:
- Llama-3.1-8B: `--model "meta-llama/Llama-3.1-8B"`
- Llama-3.2-1B: `--model "meta-llama/Llama-3.2-1B"`
- Llama-3.2-3B: `--model "meta-llama/Llama-3.2-3B"`
- Qwen-2.5-7B: `--model "Qwen/Qwen2.5-7B"` (currently only supported on N300)
- Qwen-2.5-72B: `--model "Qwen/Qwen2.5-72B"` (currently only supported on T3K)
- DeepSeek-R1-Distill-Llama-70B: `--model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`

The command to run Llama70B on Galaxy is:

```sh
MESH_DEVICE=TG LLAMA_DIR=<path to weights> TT_LLAMA_TEXT_VER="llama3_70b_galaxy" python examples/offline_inference_tt.py --model "meta-llama/Llama-3.1-70B-Instruct" --override_tt_config '{"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D", "worker_l1_size": 1344544, "trace_region_size": 62000000}'
```

### Llama-3.2 (11B and 90B) and Qwen-2.5-VL (32B and 72B) Vision models

To generate tokens (Llama-3.2-11B on N300) for sample prompts:

```sh
MESH_DEVICE=N300 python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --multi_modal --max_seqs_in_batch 16 --num_repeat_prompts 8
```

To measure performance (Llama-3.2-11B on N300) for a single batch (with the default prompt length of 128 tokens):

```sh
MESH_DEVICE=N300 python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --measure_perf --multi_modal --max_seqs_in_batch 16
```

> **Notes:**
> - To run the 11B Llama-3.2 model on QuietBox, set `MESH_DEVICE=T3K` and `--max_seqs_in_batch 32`.
> - To run the 90B Llama-3.2 model, set `MESH_DEVICE=T3K`, `--model "meta-llama/Llama-3.2-90B-Vision-Instruct"` and `--max_seqs_in_batch 4`.
> - To run the 32B Qwen-2.5-VL model, set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-32B"` and `--max_seqs_in_batch 32`.
> - To run the 72B Qwen-2.5-VL model, set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-72B"`, `--max_seqs_in_batch 32`, and `--override_tt_config '{"trace_region_size": 28467200}'`.

## Running the server example

To start up the server:

```sh
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=T3K python examples/server_example_tt.py
```

> **Notes:**
> - By default, the server will run with Llama-3.1-70B-Instruct. To run with other models, set `MESH_DEVICE` and `--model` as described in [Running the offline inference example](#running-the-offline-inference-example).
> - Custom TT options can be set using `--override_tt_config` as described in [Running the offline inference example](#running-the-offline-inference-example).

To send a request to the server:

```sh
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32, "temperature": 1, "top_p": 0.9, "top_k": 10 }'
```

### Llama-3.2 (11B and 90B) and Qwen-2.5-VL (32B and 72B) Vision models

First, start the server following the instructions above with the correct model through `--model`.

Second, generate a prompt json, e.g.,

```python
import base64
import json

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "pasta.jpeg"

# Getting the base64 string
base64_image = encode_image(image_path)

payload = {
    "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is for dinner?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 128,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
}

# Save to a JSON file
with open("server-instruct-mm-prompt.json", "w") as json_file:
    json.dump(payload, json_file, indent=4)
```

> **Notes:**
> - Qwen-2.5-VL models can also work with a real url like `"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"` instead of `"data:image/jpeg;base64,{base64_image}"`.

Finally, send a request to the server:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary @server-instruct-mm-prompt.json
```
