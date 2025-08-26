## Vision-Language Model (VLM) setup and usage

### 1) Create and activate a virtual environment
```bash
cd /path/to/your/workspace
python -m venv .vllm
source .vllm/bin/activate
pip install -U pip setuptools packaging wheel ninja 'cmake<4'
```

### 2) Clone and install vLLM (local editable)
```bash
git clone <your-vllm-fork-or-branch-url> vllm
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install -U 'transformers<4.54' timm open_clip_torch
pip install mamba-ssm --no-build-isolation
```

### 3) Prepare your HF Nano V2 VLM checkpoint
We assume you already have a Hugging Face-format checkpoint. Adjust its `config.json` for compatibility.

#### Required edits to `config.json`
- Set the top-level model type to `NemotronH_Nano_VL`.
- If present, rename the top-level key `llm_config` to `text_config`.
- Expand `text_config.hybrid_override_patterninto a full layer block type specification using the helper script below.

Minimal example of the relevant fields in `config.json`:
```json
{
  "model_type": "NemotronH_Nano_VL",
  "text_config": {
    "hybrid_override_pattern": "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-",
  }
}
```

#### Convert `hybrid_override_pattern` to full layer block spec
Run the helper script (file: `convert_layers_block_type.py`).
```bash
python3 vllm/convert_layers_block_type.py \
  --config-path=/path/to/HF_checkpoint/config.json \
  --dump-full-config
```
This above command will dump (to stdout) a patched version of the original
config.json;
the patched config which will be compatible w/ our VLLM branch.
Then replace the original config.json with the patched config dumped by the
above command.

The resulting `config.json` will be compatible with this vLLM branch. 


### 4) Notes on model/config classes (if you maintain custom code)
You need to align naming with the expected model type  in files `configuration.py` / `modeling.py`. 

- Set the config class and `model_type` in `configuration.py` as `NemotronH_Nano_VL_Config`
- If `llm_config` appears in `configuration.py`, refactor it to use `text_config` instead.
- Fix imports in `modeling.py`:

    Rename any references from `NemotronH_Nano_VL_V2_Config` to `NemotronH_Nano_VL_Config`:
    ```python
    # before
    from .configuration_something import NemotronH_Nano_VL_V2_Config as ModelConfig

    # after
    from .configuration_something import NemotronH_Nano_VL_Config as ModelConfig
    ```
- Refer to `vllm/vlm_ckpt_examples` for example `config.json` and `configuration.py` files.

### 5) Usage example 
An example script is available at `vllm/play_vlm.py`. 