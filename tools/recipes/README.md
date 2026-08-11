# vLLM Recipes Tools

Utilities for consuming deployment configurations from [vLLM Recipes](https://recipes.vllm.ai/) and converting them into files that can be used directly with vLLM.

## `recipe_json_to_vllm_config.py`

Converts a hardware-specific vLLM Recipes JSON rendering into:

- `config.yml` — native configuration for `vllm serve --config`
- `env.sh` — environment variables required by the recipe

## Install Dependency

```bash
pip install pyyaml
```

## Interactive Recipe Discovery

If you do not know the recipe JSON URL, run the script without an input:

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py
```

The script will:

1. Search the Recipes model index.
2. Ask which model to use.
3. Show the hardware configurations available for that model.
4. Resolve the correct hardware-specific JSON endpoint.
5. Generate `config.yml` and `env.sh`.

Example interaction:

```text
No recipe JSON supplied; starting Recipes API discovery.
Model search (for example: llama 3.1): llama 3.1

Matching models:
  [1] meta-llama/Llama-3.1-8B-Instruct — Llama-3.1-8B-Instruct [Meta]
Select model: 1

Available hardware:
  [1] h100
  [2] h200
  [3] trillium
  [4] xeon6
Select hardware: 4

Resolved recipe:
  Model:    meta-llama/Llama-3.1-8B-Instruct
  Hardware: xeon6
  JSON:     https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct/hw/xeon6.json
```

## Non-Interactive Recipe Discovery

For automation or CI, specify the model and hardware directly:

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6
```

The script verifies that the requested hardware rendering exists before using it.

## Direct Recipe JSON Input

If you already know the recipe JSON URL:

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct/hw/xeon6.json
```

A local JSON file is also supported:

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py recipe.json
```

## Start vLLM

After generating the files:

```bash
source env.sh
vllm serve --config config.yml
```

Example generated `config.yml`:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
tensor-parallel-size: 1
```

If the selected recipe does not require environment variables, `env.sh` will contain no additional exports.

## Custom Output Files

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --config-out llama31-xeon6.yml \
  --env-out llama31-xeon6-env.sh
```

## Scope

The converter currently targets recipes that resolve to a single `vllm serve` process.

Multi-node, disaggregated prefill/decode, or other multi-process deployment recipes require multiple commands and cannot be represented by a single `config.yml`.
