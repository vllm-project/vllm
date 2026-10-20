---
name: model-recipes
description: Find vLLM deployment recipes and configs for LLMs and Hugging Face models by model and hardware. Use for recipe lookup, creation, and updates.
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
   for supported selections. Return commands as published, including their
   flags and parallelism settings.
3. Recipes are starting points for deployment and may not be tuned for your
   workload. Check each recipe for its verified configurations, and use
   benchmarks of your workload to guide further tuning.
4. To create or update a recipe, follow the
   [authoring workflow below](#create-or-update-a-recipe).

## Find a recipe

Read recipes from the published JSON on recipes.vllm.ai. It is generated from
the source YAML with variant, hardware, and strategy settings already
resolved; the YAML contains comments and conditional overrides, so do not use
it for lookup. When presenting results, link the website page. Read the source
YAML only when creating or updating a recipe.

Each recipe has a page at `https://recipes.vllm.ai/<org>/<repo>` and JSON at
the same path plus `.json`. Paths use the full Hugging Face ID, preserving
case; do not substitute `meta.slug` or a display title. For example,
`Qwen/Qwen3.8-27B` maps to
[recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

Recipe JSON is often over 20 KB, mostly the `guide`. Instead of printing it,
read only what the request needs: each step below returns the paths for the
next, so stop once the request is answered. Commands need `curl` and `jq`.
[scripts/read_recipe.sh](scripts/read_recipe.sh) is in this skill's directory;
the examples use its path from the vLLM repository root.

### Search the catalog

Skip this step if the exact HF ID is known. Otherwise search by model ID,
family, or provider using case-insensitive keywords. Each
whitespace-separated word must occur in the ID, title, or provider:

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
recipe: fetch its own `json` to get that checkpoint's command. Its page may
belong to the parent, so link the returned `url` rather than assuming every
checkpoint has a separate page.

If there is no exact match, search the family and inspect `variants` before
concluding the recipe is missing.

### Summarize the recipe

Save the recipe JSON to a file and summarize it:

```bash
read_recipe=.agents/skills/model-recipes/scripts/read_recipe.sh
recipe=$(mktemp)
curl -fsSL https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json -o "$recipe"
"$read_recipe" "$recipe"
```

`default` is the recommended hardware, strategy, and variant; `hardware` maps
each hardware ID to its command path; `recipe_verified_hardware` lists hardware
where the recipe as a whole was verified end to end. For capability
requirements, such as tool calling or text-to-video, check `features` and
`omni_tasks`, then confirm in the guide.
Some recipes, including Omni, lack `hardware`; skip hardware selection and
use the guide's launch and task-specific sections.

### Select hardware and strategy

Match a hardware ID to the user's GPU and fetch its path from `hardware`,
prefixed with `https://recipes.vllm.ai`, rather than constructing a URL. That
response is the command object itself: single-node objects carry `command`,
and multi-node objects carry `head_command` and `worker_commands` instead, so
list its keys before selecting fields.
Check `hardware_profile` to confirm that the selected configuration matches
the user's GPU model and specifications. Its `gpu_count` describes the profile,
not necessarily the number of GPUs used by the command. Check the command's
parallelism settings, `node_count`, and all `worker_commands` against the
user's available topology, and compare the variant's `vram_minimum_gb`, an
estimate, with the total memory of the GPUs the command uses; a configuration
below that minimum is not a working option unless the guide says it was
tested. Ask when the user's hardware is unclear. If no
published command or guide command fits, report the mismatch and return the
closest one unchanged; list any adjustments the user's setup would need
separately and mark them unverified instead of editing the command.

For another strategy, follow `alternatives` in the same response. These are
static files; query parameters do not change their configuration. For other
feature/topology selections, use the recipes repository's command synthesis.

### Read deployment guidance

Read the guide sections that apply to the selection, such as prerequisites,
launch notes for the selected hardware or variant, troubleshooting, and known
limitations. Pass entries from `guide_sections` or prefixes of their text.
The script prints every matching section with its subsections and fails when
a section matches nothing:

```bash
"$read_recipe" "$recipe" prerequisites troubleshooting
```

Query other fields with `jq` on the saved file, such as `features.<name>`,
`hardware_overrides`, or `omni.tasks` for request examples.

### Return the result

Return the website link, selected checkpoint and hardware, the requested
command, and relevant prerequisites and caveats from the guide. Pick one
command and copy it, including its Docker form, exactly as the command object
or guide section gives it; do not combine commands from different sections or
add, drop, or reorder flags. List optional changes separately.

For each command, state its source, either a generated hardware or strategy
configuration or a named guide section, and whether that source says it was
tested. A `recipe_verified_hardware` entry covers the recipe on that
hardware, not every variant, strategy, or guide command, so it does not make a
command verified. Generated configurations are not automatically verified.

## Create or update a recipe

Recipes cover official or widely used checkpoints, with configurations verified
end to end on real hardware. Do not add recipes for arbitrary community
uploads, untested hardware claims, or non-standard flags that only work for one
setup. Before opening a pull request, confirm that the user ran the
configuration and can share the results.

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
in a checkout of `vllm-project/recipes` or the user's intended fork.
