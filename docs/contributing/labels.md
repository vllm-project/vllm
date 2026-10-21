# Labels

vLLM applies most labels automatically. Three systems do the work:

| | Config | Applies to |
| --- | --- | --- |
| Mergify | `.github/mergify.yml` | pull requests, by changed file path or title |
| GitHub Action | `.github/workflows/issue_autolabel.yml` | issues, by title keyword |
| Issue templates | `.github/ISSUE_TEMPLATE/*.yml` | issues, by which template was used |

A label that none of them applies has to be applied by hand, and in practice
that means it mostly doesn't get applied at all.

A template's `labels:` field must name a label that already exists — GitHub
silently skips one that doesn't, with no error anywhere.

`mergify.yml` also drives reviewer auto-assignment; see
[Collaboration](../governance/collaboration.md) for adding yourself as a
maintainer of an area.

## Adding a label

A new label needs three things. If it can't have all three, don't create it.

1. **An audience.** Someone has to want to filter on it. "It would be nice to
   know" isn't enough.
2. **An owner.** A person or SIG who triages that queue. Without one the label
   accumulates issues nobody reads.
3. **A rule, in the same PR.** If you can't write a precise rule, the label
   depends on humans remembering it exists.

## Measure before you argue

Volume claims should come from the repo, not from intuition. vLLM squash-merges,
so one commit on `main` is one PR:

```bash
git log --since="6 months ago" --oneline -- <path> | wc -l
```

For reference, at the time of writing: `speculative-decoding` 81, `llama` 85,
`mistral` 72, `tpu` 14. A subsystem in that range clears the bar comfortably.
Something in the low teens needs a stronger argument than volume.

The same applies to claiming a rule is too broad. Check what it actually labels
before rewriting it:

```bash
gh pr list --repo vllm-project/vllm --state merged --limit 200 \
  --json number,title,labels
```

Several rules in this repo have been called over-broad on inspection and turned
out to be accurate when measured.

## Writing conditions

**`files=` is exact equality. `files~=` is a regex.** Writing a pattern after
`=` compares the pattern text to each filename, so it can never match:

```yaml
- files=^examples/features/speculative_decoding/   # never fires
- files~=^examples/features/speculative_decoding/  # correct
```

**Anchor file patterns.** An unanchored fragment matches far more than intended
— `files~=cuda` also matches `requirements/cuda.txt`.

**Watch for names that contain other names.** `midashenglm.py` contains "glm"
but is unrelated to GLM, so the `glm` rule anchors to the start of the filename.

**Keep an eye on where model code lives.** Newer models live in
`vllm/models/<model>/` rather than `vllm/model_executor/models/`. A rule that
only knows the old location silently stops matching.

**Prefer titles for issue keywords.** Issue bodies contain pasted `collect_env`
output, tracebacks and configs that mention unrelated hardware and libraries.
Body matching tags the reporter's environment rather than the topic.

**Single words go in `keywords`, phrases in `substrings`.** `keywords` matches on
word boundaries, which stops a term firing inside a longer word but also misses
ordinary variations — `structured output` will not match "structured outputs".

## Keeping rules honest

`tools/pre_commit/check_label_rules.py` fails if any file condition in
`mergify.yml` matches nothing in the tree. It runs via pre-commit when that file
changes. This catches conditions that were correct when written and rotted when
the code moved.

## Not everything needs a label

Triage and workflow labels — `bug`, `stale`, `ready`, `RFC`, `needs-rebase` —
and the bot provenance tags are applied by other mechanisms or by hand, and are
intentionally outside this system.
