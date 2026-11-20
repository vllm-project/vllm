# Laya decision model

`LayaForDecision` serves `convaiinnovations/laya` as a pooling model. Its
ModernBERT encoder uses bidirectional full and sliding attention; the decision
head scores the option marker tokens. It does not generate text.

## Prerequisites

`LayaForDecision` targets vLLM `main` (post-`v0.32.0`); older release builds
such as `v0.32.0rc1` do not include this model architecture or newer core
APIs. On Ascend, pair this tree with a matching `vllm-ascend` `main` checkout
whose `.github/vllm-main-verified.commit` aligns with the vLLM baseline.

If you overlay or mount a post-`v0.32.0` vLLM source tree onto an older
trimmed container image instead of building a fresh main image:

1. **Additional Python dependencies**: vLLM `main` imports runtime packages
   listed in `requirements/common.txt` that may be absent from older images
   (notably `orjson>=3.11.6`, imported by the API server utilities). Install
   missing pure-Python packages with `--no-deps` (for example,
   `pip install --no-deps orjson`) rather than running an unrestricted
   reinstall that could upgrade `torch` or break binary compatibility.
2. **Ascend version gate (`VLLM_VERSION`)**: When `vllm` package metadata in
   the container remains at `0.32.0*` (or falls back to `dev`), `vllm-ascend`
   version checks such as `vllm_version_is("0.32.0")` can route imports to
   removed release paths (for example `kv_cache_utils.create_kv_cache_group_specs`
   instead of `kv_cache_planning`). Export a PEP 440 `main`-lane version
   string that matches neither `0.32.0` nor `0.33.0` before both
   `vllm-ascend` installation and `vllm serve`:

   ```bash
   export VLLM_VERSION=0.32.1.dev1
   ```

## Prepare the checkpoint

The original repository stores encoder configuration and tokenizer files in
subdirectories. Create a serving view without modifying the weights. The
following script expects an already downloaded checkpoint in `./laya` and
creates `./laya-serving`:

```python
import json
from pathlib import Path

from transformers import AutoTokenizer

root = Path("laya").resolve()
target = Path("laya-serving").resolve()
target.mkdir(exist_ok=True)
settings = json.loads((root / "rl_agent_config.json").read_text())
config = json.loads((root / "encoder/config.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
config.update({
    key: settings[key]
    for key in ("head_layers", "temperature", "temperature_by_options")
})
config.update(
    architectures=["LayaForDecision"],
    dtype="bfloat16",
    n_act=len(settings["act_costs"]) + 1,
    mask_token_id=tokenizer.mask_token_id,
    type_token_ids={
        kind: tokenizer(kind + " question:", add_special_tokens=False)["input_ids"][0]
        for kind in ("choice", "score", "noul")
    },
)
(target / "config.json").write_text(json.dumps(config, indent=2))
for source in [root / "model.safetensors", *(root / "tokenizer").iterdir()]:
    destination = target / source.name
    if source.is_file() and not destination.exists():
        destination.symlink_to(source)
```

## Serve and query

```bash
vllm serve ./laya-serving --served-model-name laya --runner pooling \
  --pooler-config '{"task":"classify","use_activation":true}' \
  --dtype bfloat16 --max-model-len 512 --max-num-batched-tokens 512 \
  --max-num-seqs 1 --no-enable-prefix-caching --no-enable-chunked-prefill \
  --enforce-eager
```

The example above uses a 512-token budget; `--max-model-len` and
`--max-num-batched-tokens` may be raised up to the encoder configuration's
`max_position_embeddings` (keeping `--max-num-batched-tokens` at least
`--max-model-len` while chunked prefill is disabled).

Use the checkpoint's `rl_common.build_sequence` to construct prompt token IDs
for one question, including its type and option markers. Send those IDs as
`input` to `/classify`. The returned vector contains the option probabilities
followed by the action probabilities. The decision head runs in FP32; the
encoder can run in BF16. Preserve the checkpoint's per-type and per-option
count temperatures when comparing against its original `RLAgent` reference.

Graph execution depends on the platform's encoder attention implementation.
Ascend validation covers Eager and PIECEWISE with six short/long text cases
against the original CPU FP32 reference; it does not establish leaderboard
accuracy or multimodal support.
