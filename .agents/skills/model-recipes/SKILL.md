---
name: model-recipes
description: Find community-maintained vLLM recipes by model and hardware. Use for recipe lookup, creation, and updates.
---

# Model Recipes

Model serving recipes are maintained in the
[models/ directory of vllm-project/recipes](https://github.com/vllm-project/recipes/tree/main/models).
The published catalog is at
[recipes.vllm.ai](https://recipes.vllm.ai).

## Usage notes

1. Recipe lookup returns commands and deployment guidance without launching
   `vllm serve`.
2. Use published configurations or the recipes repository's command generator
   for supported selections. During lookup, preserve the resulting flags and
   parallelism settings.
3. Recipes are starting points for deployment and may not be tuned for your
   workload. Check each recipe for its verified configurations, and use
   benchmarks of your workload to guide further tuning.
4. To create or update a recipe, follow the
   [authoring workflow below](#create-or-update-a-recipe).

## Find a recipe

### Choose a source

Use either of these recipe sources:

- **Website:** [recipes.vllm.ai](https://recipes.vllm.ai), with model pages at
  `https://recipes.vllm.ai/<org>/<repo>` and published JSON at the same path
  plus `.json`. [Browse](https://recipes.vllm.ai/browse) supports human browsing;
  agents can query the JSON resources below without HTML scraping.
- **Original files:** [models/](https://github.com/vllm-project/recipes/tree/main/models)
  contains recipe YAML at `models/<org>/<repo>.yaml`. Read it through GitHub,
  a local checkout, or
  `https://raw.githubusercontent.com/vllm-project/recipes/main/models/<org>/<repo>.yaml`.
  Read the complete YAML, including `guide`, dependencies, and verification
  notes. Use the published JSON or the repository's command generator when a
  generated hardware/strategy command is needed.

Model paths follow the full Hugging Face ID (`<org>/<repo>`), preserving case;
do not substitute `meta.slug` or a display title. For example,
`Qwen/Qwen3.8-27B` maps to the
[website page](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) and
[source YAML](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml).

### Search by model

Search by model ID, family, or provider using case-insensitive keywords.
Each whitespace-separated word must occur in the ID, title, or provider:

```bash
curl -fsSL https://recipes.vllm.ai/models.json |
  jq --arg q 'qwen 27b' '
    ($q | ascii_downcase | [scan("\\S+")]) as $words | .[] |
    ([.hf_id, .title, .provider] | join(" ") | ascii_downcase) as $text |
    select(all($words[]; . as $word | $text | contains($word))) |
    {hf_id, json, url, derived_from}'
```

`json` is the recipe data path; `url` is its page path. Prefix both with
`https://recipes.vllm.ai`. A `derived_from` entry is a variant of a parent
recipe: fetch its own `json` to get that checkpoint's command.
Its page and source YAML may belong to the parent: follow the returned `url`
and use `derived_from` to locate the parent YAML and its `variants`, rather
than assuming every checkpoint has a separate source file.

If there is no exact match, search the family and inspect `variants` before
concluding the recipe is missing.

### Read the recipe and deployment guidance

Fetch the returned `json` path or read the source YAML. If the exact HF ID is
known, try `/<org>/<repo>.json` directly, preserving its case:

```bash
curl -fsSL https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json
```

Read `guide` alongside the command for prerequisites, deployment tips, and
limitations. Keep `features`, `hardware_overrides`, and `meta.hardware`
available when selecting a configuration; a command-only summary omits them.
For specific capability requirements, such as tool calling or text-to-video,
check `features`, `omni.tasks`, and `guide` in the JSON or YAML.

### Select hardware and strategy

`recommended_command` contains the default hardware, strategy, and command.
List `recommended_command.by_hardware | keys` to discover hardware IDs, then
fetch the selected entry's returned path rather than constructing a URL.
That response is the command object itself. Check `hardware_profile` to confirm
that the selected configuration matches the user's GPU model and specifications.
Its `gpu_count` describes the profile, not necessarily the number of GPUs used
by the command. Check the command's parallelism settings,
`node_count`, and all `worker_commands` against the user's available topology.
Ask when the user's hardware is unclear; if no published configuration fits,
report that mismatch.

For another strategy, follow `alternatives` in the selected hardware's response.
These are static files;
query parameters do not change their configuration. For other feature/topology
selections, use the recipes repository's command synthesis. Some recipes,
including Omni, lack `recommended_command`; use their `guide` and task-specific
instructions instead of the hardware/strategy JSON workflow.

### Return the result

Return the website and source YAML links, selected checkpoint and hardware,
the requested command, and relevant
prerequisites and caveats from the guide. Report the verification scope recorded
in the guide and `meta.hardware`; generated configurations are not automatically
verified.

## Create or update a recipe

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
in a checkout of `vllm-project/recipes` or the user's intended fork.
