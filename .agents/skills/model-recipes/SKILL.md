---
name: model-recipes
description: Find community-maintained vLLM recipes for running a model on specific hardware for a given task. Use for recipe lookup, creation, and updates.
---

# Model Recipes

Model serving recipes are maintained in
[vllm-project/recipes](https://github.com/vllm-project/recipes), separately from
this vLLM repository. The published catalog is at
[recipes.vllm.ai](https://recipes.vllm.ai).

## Find a recipe

Query the published catalog directly; no clone is needed. For browsing, use
[Browse](https://recipes.vllm.ai/browse); for command retrieval, use the JSON
resources below.

**Search by model ID, family, or provider** with a case-insensitive substring:

```bash
curl -fsSL https://recipes.vllm.ai/models.json |
  jq --arg q 'qwen3.8-27b' '.[] |
    select([.hf_id, .title, .provider] | join(" ") | ascii_downcase |
      contains($q | ascii_downcase)) |
    {hf_id, json, url, derived_from}'
```

`json` is the recipe data path; `url` is its page path. Prefix both with
`https://recipes.vllm.ai`. A `derived_from` entry is a variant of a parent
recipe: fetch its own `json` to get that checkpoint's command.

**Search by task** using the Browse page's task filter, for example
[multimodal recipes](https://recipes.vllm.ai/browse?task=multimodal).
The categories are `text`, `multimodal`, `omni`, and `embedding`.
`models.json` does not contain task labels; inspect `meta.tasks` in a matched
recipe's JSON. For finer requirements, inspect `features` (such as reasoning
or tool calling), `omni.tasks` (such as text-to-video), and the guide. A task
label describes a category, not evidence of task-specific tuning.

**Read a matched recipe** using the returned `json` path. If the exact HF ID
is known, try `/<org>/<repo>.json` directly, preserving its case:

```bash
curl -fsSL https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json |
  jq '{tasks: .meta.tasks, model, variants, recommended_command}'
```

`recommended_command` contains the default hardware, strategy, and command.
For another GPU, fetch its path in `recommended_command.by_hardware`; that
response is the command object itself. For another strategy, follow
`alternatives` in the selected hardware's response. These are static files;
query parameters do not change their configuration. Check `node_count` and all
`worker_commands` before reusing a multi-node result. For other feature/topology
selections, use the page's command builder or the recipes repository's command
synthesis. Some recipes, including Omni, lack `recommended_command`; read their
`guide` and task-specific instructions instead.

If there is no exact match, search the family and inspect `variants` before
concluding the recipe is missing. Return the page link, selected checkpoint and
hardware, and the requested command. Distinguish generated configurations from
hardware actually marked verified in the recipe.

## Create or update a recipe

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
in a checkout of `vllm-project/recipes` or the user's intended fork.
