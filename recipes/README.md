# vLLM model recipes

This directory holds the deployment recipes for models served by vLLM, one
YAML per model at `models/<hf_org>/<hf_repo>.yaml`. The path mirrors the
Hugging Face ID, so `models/Qwen/Qwen3.8-27B.yaml` renders at
[recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

The site, the command builder, and the JSON API are built by
[vllm-project/recipes](https://github.com/vllm-project/recipes). That
repository owns the schema and the shared vocabulary (hardware profiles,
serving strategies, KV-offload deployments) and pins this directory to a
commit. Omni (image, video, audio) recipes stay in vllm-project/recipes for
now (moving to vllm-omni), and Ascend NPU content is inline in these recipes
(moving to vllm-ascend overlays).

## Editing a recipe

Start a new recipe from
[`templates/recipe.yaml`](https://github.com/vllm-project/recipes/blob/main/templates/recipe.yaml).
Fields are defined in
[SCHEMA.md](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md) and
checked against [`recipe.schema.json`](https://recipes.vllm.ai/recipe.schema.json)
on every build; an unknown or misspelled key is an error. For completion and
inline errors in YAML-aware editors, start the file with
`# yaml-language-server: $schema=https://recipes.vllm.ai/recipe.schema.json`.

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
for the schema and the validation steps. To validate a change exactly as this
repository's `Recipes` PR check does, run from a checkout of
`vllm-project/recipes`:

```bash
node scripts/validate-source.mjs --source vllm --dir ~/vllm/recipes
```

To preview the site with your changes, add a gitignored `sources.local.yaml`
to that checkout with `overrides: { vllm: ~/vllm/recipes }` and run `pnpm dev`.
This works once vllm-project/recipes reads this directory as its `vllm`
source (vllm-project/recipes#1095).

A pull request that changes this directory gets the `recipe` label and a
comment linking its own preview deployment, rebuilt a few minutes after each
push.
