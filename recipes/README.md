# vLLM model recipes

This directory holds the deployment recipes for models served by vLLM, one
YAML per model at `models/<hf_org>/<hf_repo>.yaml`. The path mirrors the
Hugging Face ID, so `models/Qwen/Qwen3.8-27B.yaml` renders at
[recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

The site, the command builder, and the JSON API are built by
[vllm-project/recipes](https://github.com/vllm-project/recipes). That
repository owns the schema and the shared vocabulary (hardware profiles,
serving strategies, KV-offload deployments) and pins this directory to a
commit. Recipes from other repositories are merged in at build time:
vllm-omni owns the omni (image, video, audio) recipes, and vllm-ascend adds
Ascend NPU content through overlays.

## Editing a recipe

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
for the schema and the validation steps. To preview a change, point a
checkout of `vllm-project/recipes` at this directory with a gitignored
`sources.local.yaml`:

```yaml
overrides:
  vllm: ~/vllm/recipes
```

Then run `node scripts/build-recipes-api.mjs` (validation and JSON API) or
`pnpm dev` (site) in that checkout.
