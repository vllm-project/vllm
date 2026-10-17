---
name: transformers-upgrade
description: "Playbook for changing vLLM's Transformers version bounds. Use when bumping the pinned or maximum Transformers version, raising the minimum Transformers version, or deleting vendored configs, processors, or compat shims that upstream Transformers now provides."
---

# Upgrading Transformers

The supported range lives in `requirements/common.txt`. CI pins one exact
version in `requirements/test/*.in` and the compiled `requirements/test/*.txt`.
A version change is one of two jobs, and they usually come as a pair:

- **Bump the pin**: a new Transformers release becomes the CI version and
  the upper bound. vLLM grows compat code for it.
- **Raise the floor**: the minimum moves up. That compat code, and every
  vendored config or processor upstream now ships, becomes dead weight.
  Every floor raise ends with an **unvendoring pass**.

## Bump the pin

1. Edit the pin line in each `requirements/test/*.in` and `*.txt` and the
   upper bound in `requirements/common.txt`. Past bumps edit only the pin
   line in the compiled files; see `git log -- requirements/common.txt`.
2. Read the release notes for config validation changes, renamed auto-mapping
   formats, and removed attributes; these are what break vLLM.
3. Search `tests/` for `min_transformers_version` and skip markers naming the
   new version, and un-gate them.
4. Run the suites in [Testing](#testing) and fix what breaks. Mark every
   shim with a `# Transformers >= X` comment so the next floor raise finds it.

Done when the [Testing](#testing) suites pass on the new pin and every test
waiting on this version is un-gated.

## Raise the floor (unvendoring pass)

### 1. Inventory

Inventory candidates with the floor version installed (no vLLM install needed):

```bash
uv run --no-project --with transformers==<floor> \
    python .agents/skills/transformers-upgrade/scripts/find_upstreamed.py
```

It lists:

- vendored configs whose `model_type` upstream now owns;
- `_CONFIG_REGISTRY` overrides of upstream types;
- registered processors that collide with upstream class names or models;
- version-gated code at or below the floor;
- code that reads the installed Transformers version, whatever its threshold;
- patches citing a Transformers issue or PR (with whether a referenced PR is
  in the floor release) or a future minimum version.

It finds candidates by name only. Every hit still needs a verdict.

### 2. Port each candidate to upstream

Every hit is deleted by default. A vendored copy is a liability: it drifts
from upstream, hides upstream fixes, and shadows the native class. Port the
vLLM model code to the upstream schema instead, renaming attribute reads and
deriving values from upstream fields.

Fan candidates out to parallel subagents. Each one compares the vendored and
upstream classes, then greps every attribute the vLLM model code reads.
[Upstreaming hazards](#upstreaming-hazards) lists the usual fixes.

Keep a vendored class only when upstream cannot load a checkpoint vLLM must
keep serving, and record why in the PR description. A different upstream
schema, a model-code rewrite, or a needed eval is work for the PR, not a
reason to keep.

### 3. Prove it on real checkpoints

For every checkpoint in `tests/models/registry.py` that uses a ported
config, dump what vLLM builds before and after the change, then diff the two.
Run both dumps; each catches what the other misses:

- `dump_configs.py`: the resolved config, attribute by attribute.
- `dump_model.py`: the model built on the meta device with no GPU or weights.
  It records ModelConfig sizes, every parameter shape, and every module's
  scalar attributes (`head_dim`, `num_kv_heads`, `scaling`, ...). Missing
  fields that are read with defaults only show up here.

```bash
git worktree add --detach <scratch>/vllm-base upstream/main
S=.agents/skills/transformers-upgrade/scripts
for dump in dump_configs dump_model; do
    PYTHONPATH=<scratch>/vllm-base .venv/bin/python $S/$dump.py before.jsonl <repos>
    PYTHONPATH=$PWD .venv/bin/python $S/$dump.py after.jsonl <repos>
    .venv/bin/python $S/diff_dumps.py before.jsonl after.jsonl
done
```

Add `--trust-remote-code` for checkpoints whose `config.json` has an
`auto_map`, and dump both ways.

How to read the diffs:

- **A `<MISSING>` config field the model reads:** this crashes or silently
  falls back to a default. Port the model code to the upstream field.
- **New upstream-only defaults** (`attention_bias=False`,
  `pad_token_id=None`): usually benign. A model diff of 0 confirms it.
- **Any model diff:** this is a numerics change, and the PR needs a model
  eval. For example, A.X-K1 gained the checkpoint's YaRN rope scaling, which
  the vendored class had been dropping; it shows up as
  `layers.*.self_attn.scaling (x61)`.

### 4. Delete

For each deleted config:

- Delete the file.
- Remove its entries from `_CLASS_TO_MODULE` and `__all__` in
  `vllm/transformers_utils/configs/__init__.py`.
- Remove its entries from `_CONFIG_REGISTRY` in
  `vllm/transformers_utils/config.py`.
- Switch every importer, including tests, to `from transformers import ...`.
- Remove the `source_file_dependencies` lines naming the file from
  `.buildkite/`.

Deleted processors follow the same steps in
`vllm/transformers_utils/processors/__init__.py`.

Done when every line of the inventory output is deleted, or kept with a
reason that meets the bar in step 2. The PR also carries the config diff
results and evals for any changed values.

## Upstreaming hazards

Upstream configs are `@strict` dataclasses. Checkpoint keys upstream does not
declare survive as plain attributes. Defaults that only vLLM sets, and values
vLLM derives, vanish.

| Hazard | Signal | Fix |
| --- | --- | --- |
| Upstream consumes legacy keys | `__post_init__` pops keys the model reads (`deepseek_v4` `compress_ratios`) | Read the upstream replacement (`layer_types`, `mlp_layer_types`, per-layer `rope_parameters`) |
| Renamed or dropped fields | Diff shows `<MISSING>` for a field the model reads | Read the upstream name; derive dropped flags from what implies them (`enable_moe_block` from `num_experts`) |
| Type check on the config | `ctx.get_hf_config(SomeConfig)` relies on the vendored base class | Accept the upstream class too, e.g. a `Cosmos3ProcessingInfo` overriding `get_hf_config` |
| Remote code takes over | Checkpoint `config.json` has `auto_map`; with `--trust-remote-code`, the hub class replaces the native one once vLLM stops registering its own | Check the remote class loads; if not, register the upstream class for that `model_type` |
| Read-only upstream fields | Properties or derived fields break subclasses and `hf_overrides` (`nemotron_h` `hybrid_override_pattern`) | Override the source field (`layers_block_type`) instead |
| Behavioural rewrite | Upstream derives fields differently (`diffusion_gemma` `sliding_window`) | Upstream matches the reference implementation; verify with an eval |
| Original checkpoint format | Upstream targets `-hf` conversions; vLLM serves the originals (`qwen3_asr`, `qianfan_ocr`) | Keep only if upstream cannot load the original at all |
| Processor output contract | Model code consumes keys upstream does not emit | Port the model to upstream's keys |
| Name collision | A registered vendored processor shares an upstream class name and hijacks `get_processor` (`InternVLProcessor`) | Drop the registry entry |

Version gates from the inventory need a decision:

- **Code** only reachable below the floor is deleted. This covers:
    - comparisons against `TRANSFORMERS_VERSION`;
    - branches labelled `# Transformers < X`;
    - fallbacks for fields that upstream configs now always set, e.g.
    `mlp_layer_types`, `per_layer_config` or legacy `layer_types` names.

  A `skipif` whose condition is now always true becomes an unconditional
  `skip` with the same reason.
- **`min_transformers_version`** at or below the floor is removed.
- **`max_transformers_version`** below the floor means CI never tests that
  model; report it rather than deleting it silently.
- **Patches citing a Transformers issue or PR:**
    - Delete the patch when its fix is in the floor release.
    - For an issue, find the PR that closed it and check that PR. A closed
      issue is not a fixed one: #43329 was closed while the bug was still in
      5.16.1.
    - Re-run the patched behaviour before deleting. A fix in the release does
      not always cover vLLM's path: #47924 made Emu3's processor add BOS, but
      vLLM tokenizes without the processor.
    - Links that only explain a design choice (e.g. a float32 RoPE note) are
      not patches; leave them.

## Testing

```bash
.venv/bin/python -m pytest tests/transformers_utils tests/config tests/tokenizers_
.venv/bin/python -m pytest tests/models/multimodal/processing/test_common.py -k "<models>"
```

Add `tests/models/test_initialization.py -k "<archs>"` where a GPU is
available, and a model eval for any model whose resolved config changed.
