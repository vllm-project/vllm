# Tuning `skip_softmax_threshold_scale_factor_{prefill,decode}` with NVIDIA Model-Optimizer

This guide explains how to use
[`flash_skip_softmax.py`](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/sparsity/attention_sparsity/methods/flash_skip_softmax.py)
from NVIDIA Model-Optimizer to **calibrate**, per model, the two vLLM flags
this branch adds:

- `--attention-config.skip_softmax_threshold_scale_factor_prefill`
- `--attention-config.skip_softmax_threshold_scale_factor_decode`

The calibration workflow converts a desired **target sparsity** (e.g.
"skip ~50% of attention blocks") into the concrete scale-factor numbers
that the TRTLLM/FlashInfer kernels consume.

---

## 1. Why these two numbers exist

Both vLLM and Model-Optimizer use the same block-wise skip-softmax rule.
In pseudocode (see `flash_skip_softmax.FlashSkipSoftmax.calc_correction_factor_and_p`):

```text
per-block rule:
    drop block iff   max_block_score - running_cummax  <  log(threshold)

threshold formula:
    threshold = scale_factor / seq_k
```

So:

- `scale_factor` is **sequence-length-independent**. It's the "knob".
- `seq_k` (context length) is what the kernel already knows at runtime.
- `threshold` is what the kernel actually compares against.

vLLM exposes `scale_factor` directly so the same server config behaves
consistently across ISLs. From
[`vllm/config/attention.py`](../vllm/config/attention.py):

> *"The actual threshold equals this value divided by the context length.
> Higher values increase kernel performance at the cost of accuracy."*

Model-Optimizer's calibration fits an **Exponential model** per phase:

```text
scale_factor(target_sparsity)  =  a * exp(b * target_sparsity)
```

and stores `(a, b)` as `calibration_params[phase]`. To plug this into
vLLM you just evaluate that expression at your chosen target sparsity
and pass the number to the CLI flag. No other translation is needed.

---

## 2. Prerequisites

```bash
# One-time: install uv and create the vLLM venv per AGENTS.md
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt

# vLLM (this branch) — Python-only build is enough for serving.
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Model-Optimizer with the attention-sparsity extras.
uv pip install "nvidia-modelopt[torch]>=0.25.0"

# HF model + calibration dependencies.
#   - accelerate: required for `device_map="auto"` model loading.
#   - nltk + wonderwords: RULER dataset generator uses them; without
#     them calibration raises `ModuleNotFoundError: No module named 'nltk'`.
#   - datasets / transformers: modelopt picks the currently-installed
#     versions. Calibration has been exercised against transformers 5.6
#     (see §4 for a known workaround).
uv pip install "transformers>=4.45" accelerate datasets nltk wonderwords
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
```

Calibration patches `torch.nn.functional.softmax` in HF Transformers'
**eager** attention path, so the model *must* be loadable with
`attn_implementation="eager"`. This is a Model-Optimizer requirement,
not a vLLM requirement — at serve time vLLM still uses its native
FlashInfer / TRTLLM kernels.

Recommended environment variable when calibrating at long
`max_seqlen` (≥ 16k): set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching
the calibration. Without it we observed intermittent
`CUDA error: unspecified launch failure` raised from
`torch.cuda.empty_cache()` at the end of a chunked prefill — the
symptom of a prior kernel failing async due to allocator
fragmentation, not an actual OOM.

---

## 3. Calibration script

A ready-made driver that encodes all the workarounds below lives at
[`tools/calibrate_skip_softmax.py`](../tools/calibrate_skip_softmax.py).
Typical invocation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python tools/calibrate_skip_softmax.py \
    --model /models/Qwen3/Qwen3-30B-A3B-Instruct-2507 \
    --out calibration/qwen3_30b_a3b_sparsity70.json \
    --target-sparsity-prefill 0.7 --target-sparsity-decode 0.7 \
    --samples 24 --max-seqlen 16384 --chunk-size 4096 \
    --cache-dir calibration/ruler_cache \
    --also-evaluate 0.3 0.5 0.7
```

The minimal self-contained version — what the driver boils down to —
looks like this (adjust `MODEL` to your checkpoint):

```python
import json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
import modelopt.torch.sparsity.attention_sparsity as mtsa

MODEL = "openai/gpt-oss-120b"            # any HF causal-LM id
OUT   = Path("calibration/skip_softmax.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load the model in eager mode (required by flash_skip_softmax).
#    trust_remote_code is needed for some vendor checkpoints.
# ------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# ------------------------------------------------------------------
# 2. Build the sparse-attention config with CALIBRATION enabled.
#    IMPORTANT — pattern matching:
#      modelopt fnmatches against the MODULE PATH, not the class name.
#      HF Qwen / Llama etc. name the attention attribute `self_attn`
#      (paths like `model.layers.0.self_attn`), so the naive
#      "*attention*" wildcard does NOT match them.
#      Use "*" (everything) or be explicit with "*self_attn*".
#      Auto-registration still only registers modules whose CLASS NAME
#      looks attention-ish (ends in "Attention"/"SelfAttention" or
#      contains "attention"), so "*" is safe here.
# ------------------------------------------------------------------
config = {
    "sparse_cfg": {
        "*": {
            "method": "flash_skip_softmax",
            "backend": "pytorch",
            "enable": True,
            "br": 128,
            "bc": 128,
            "is_causal": True,
        },
        "calibration": {
            "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
            "samples": 48,
            "max_seqlen": 16384,
            "chunk_size": 4096,
            "num_decode_tokens": 10,
            # "threshold_trials": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1],
        },
    },
}

# ------------------------------------------------------------------
# 3. Apply the monkey-patch from §4 BEFORE calling sparsify() if you
#    are on transformers 5.x (the default decode forward loop hard-
#    codes flash_attention_2 and crashes on Qwen3-MoE / similar).
# ------------------------------------------------------------------
mtsa.sparsify(model, config)

# ------------------------------------------------------------------
# 4. Pull (a, b) out of ONE sparse module — they are shared across
#    all attention layers by design.
# ------------------------------------------------------------------
from modelopt.torch.sparsity.attention_sparsity.utils import (
    get_named_sparse_attention_modules,
)
(_, any_mod), *_ = get_named_sparse_attention_modules(model)
method = any_mod._sparse_method_instance
params = method.calibration_params     # {"prefill": {"a":..,"b":..}, "decode": {...}}
targets = method.target_sparse_ratio   # {"prefill": 0.5, "decode": 0.5}

def scale(phase, t):
    return float(params[phase]["a"] * math.exp(params[phase]["b"] * t))

OUT.write_text(json.dumps({
    "model": MODEL,
    "calibration_params": {p: {k: float(v) for k, v in params[p].items()} for p in params},
    "target_sparse_ratio": {k: float(v) for k, v in targets.items()},
    "vllm_flags": {
        "skip_softmax_threshold_scale_factor_prefill": scale("prefill", targets["prefill"]),
        "skip_softmax_threshold_scale_factor_decode":  scale("decode",  targets["decode"]),
    },
}, indent=2))
```

Measured runtime on 4× GB200 for Qwen3-30B-A3B (30B MoE / 3B active),
`samples=24`, `max_seqlen=16384`, `chunk_size=4096`,
`num_decode_tokens=10` — **~3 min of actual calibration** on top of
model-load time (~16 s when weights are page-cached, ~16 min cold).
Dense LLMs in the same parameter range take roughly the same order of
magnitude per phase.

### Notes on the calibration config

| Field | Default | Guidance |
| ----- | ------- | -------- |
| `target_sparse_ratio.prefill` | `0.5` | Higher = faster long-context prefill, lower accuracy. `0.3–0.7` is typical. Set to `0.0` to skip prefill calibration entirely. |
| `target_sparse_ratio.decode`  | `0.5` | Decode benefits most at long KV-cache occupancy. Set to `0.0` to skip decode calibration. |
| `samples` | `24` | 24 is 1 sample × 6 RULER tasks × 4 length bins. Bump to 48–96 for production calibration, **especially if decode R² comes out < 0.8** (see §4). |
| `max_seqlen` | `32768` | Must be ≥ 1024 **and** long enough that achievable sparsity in the raw threshold sweep clearly exceeds your target. Empirically at `max_seqlen=4096` sparsity saturates around ~49%, so a 70% target is unreachable there — the fit then extrapolates and gives meaningless scale factors. As a rule of thumb, use a `max_seqlen` where the raw sweep reaches at least `target + 10%`. Pick the longest sequence length you expect at serve time if feasible. |
| `chunk_size` | `2048` | Keep ≤ your prefill memory budget. `-1` disables chunking. Chunked prefill is what triggered the allocator fragmentation we worked around with `expandable_segments:True`. |
| `num_decode_tokens` | `10` | How many tokens of decode to profile per sample. Double it to `20` if decode R² is below `0.7`. |
| `threshold_trials` | 20 trials from `1e-6` to `0.99` | Narrow this only if you know the regime you care about. |
| `fit_logspace` | `False` | Leave `False` for causal LLMs; `True` is mainly for diffusion models. |
| `cache_dir` | `None` | Set it — RULER sample generation takes ~25 s and the cache filename is keyed on `(samples, max_seqlen)`, so repeat runs reuse it. |

---

## 4. Known issues & workarounds

### 4.1 Transformers 5.x decode forward loop crashes on recent HF arches

The default decode-phase forward loop
(`create_decode_calibration_forward_loop`) hard-codes
`model.config._attn_implementation = "flash_attention_2"` for the fast
prefill that populates the KV cache before the measured decode steps.
On transformers **5.6.0** + Qwen3-MoE this crashes with:

```text
File .../transformers/integrations/flash_attention.py, line 84, in flash_attention_forward
    s_aux=s_aux.to(query.dtype),  # FA only accepts half precision
          ^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'to'
```

(the `s_aux` attention-sinks field is only populated by GPT-OSS-style
models, and the integration forgets to handle the `None` case for
everything else). The same symptom has shown up on other recent HF
causal-LM classes in 5.x.

**Workaround** — patch the factory to use `sdpa` instead of
`flash_attention_2` for the fast prefill. SDPA also does not call
`F.softmax`, so measurement is unaffected, but it is the safe,
widely-tested path in modern transformers. Apply this *before*
`mtsa.sparsify(...)`:

```python
import torch
import modelopt.torch.sparsity.attention_sparsity.calibration.calibrate as _cal
from modelopt.torch.utils import get_module_device

def _safer_decode_factory(calibration_data, tokenizer_name_or_path, num_decode_tokens=10):
    tok = _cal._load_tokenizer(tokenizer_name_or_path)

    def forward_loop(model):
        device = get_module_device(model)
        for sample in calibration_data:
            inputs = tok(
                sample["input"],
                return_tensors="pt", truncation=True, max_length=sample["length"],
            )
            input_ids = inputs["input_ids"].to(device)
            original = getattr(model.config, "_attn_implementation", "eager")
            with torch.no_grad():
                try:
                    model.config._attn_implementation = "sdpa"   # was "flash_attention_2"
                    outputs = model(input_ids, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                    del outputs

                    model.config._attn_implementation = "eager"
                    for _ in range(num_decode_tokens):
                        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                        past_kv = outputs.past_key_values
                        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                        del outputs
                finally:
                    model.config._attn_implementation = original
            del past_kv
            torch.cuda.empty_cache()

    return forward_loop

_cal.create_decode_calibration_forward_loop = _safer_decode_factory
```

This monkey-patch is already baked into
[`tools/calibrate_skip_softmax.py`](../tools/calibrate_skip_softmax.py).
Note that the public
`mtsa.sparsify(..., forward_loop=...)` only overrides the *prefill*
forward loop — the decode loop is not user-overridable via a clean
API, hence the monkey-patch.

### 4.2 Async CUDA launch failure from `torch.cuda.empty_cache()` at long seqlens

When calibrating at `max_seqlen ≥ 16384` with chunked prefill we
intermittently saw:

```text
torch.AcceleratorError: CUDA error: unspecified launch failure
  File .../calibration/calibrate.py, line 140, in forward_loop
    torch.cuda.empty_cache()
```

The error is raised at `empty_cache` but is actually an asynchronous
failure from an earlier kernel in the chunked prefill — typically
allocator fragmentation, not an honest OOM (on 192 GB GB200s we had
>150 GB free). Export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching
the calibration; with this set our 16k calibration consistently
completes. Running under `CUDA_LAUNCH_BLOCKING=1` is also useful for
localising the real culprit on a first failure.

### 4.3 `max_seqlen` below the target-sparsity saturation point

Short `max_seqlen` caps the *achievable* sparsity regardless of
threshold. Concretely, for Qwen3-30B-A3B we observed:

| `max_seqlen` | Max sparsity in raw threshold sweep |
| ------------ | ----------------------------------- |
| 4096  | ~49% |
| 16384 | ~71% (prefill) / ~95% (decode) |

If your target is 70% and you calibrate at 4096, the fit is
extrapolating beyond the observed data and the resulting scale factor
is **not meaningful**. The calibration log prints the per-threshold
sparsity sweep — always sanity-check that your target lies inside
that range before trusting the exported numbers.

### 4.4 Decode R² can stay low even on a well-converged sweep

For Qwen3-30B-A3B we got prefill R² = 0.91 but decode R² = 0.67, with
the fitted `b` clamped to the optimiser's upper bound of 20.0. This
usually means the decode sparsity–vs–threshold curve is too sharp for
a single exponential to capture (very low sparsity at small
thresholds, then a very steep climb).

Mitigations, in order of preference:

1. Increase `num_decode_tokens` to 20 and/or `samples` to 48+ —
   more points, more stable fit.
2. Cross-check the final scale factor against the raw sweep table
   printed during calibration. If the `Avg SF` at the row whose
   `Avg Sparsity` matches your target is within ~2× of the exported
   `scale_factor_decode`, the fit is consistent even with a mediocre
   R². If they disagree by an order of magnitude, distrust the fit
   and deploy a value taken directly from the raw sweep table.
3. If you ultimately don't care about decode-side skip-softmax (e.g.
   most of your traffic is short prefills with modest OSL), set
   `target_sparse_ratio.decode = 0.0` and omit the `_decode` flag.

### 4.5 Third-party runtime deps aren't in `nvidia-modelopt[torch]`

Fresh installs of `nvidia-modelopt[torch]` alone will fail at first
run with:

- `ValueError: ... requires accelerate` → `pip install accelerate`
- `ModuleNotFoundError: No module named 'nltk'` → `pip install nltk`,
  then `nltk.download('punkt')` and `nltk.download('punkt_tab')`.
- `ModuleNotFoundError: No module named 'wonderwords'` →
  `pip install wonderwords`.

All of these are in the install snippet in §2; include them in any
container / devcontainer image you build for this workflow.

---

## 5. Converting `(a, b)` to a vLLM flag

The one and only formula (the Python script above already does this):

```text
scale_factor_<phase>  =  a_<phase>  *  exp( b_<phase>  *  target_sparsity_<phase> )
```

**Real-world worked example** — the calibration run in
[§3](#3-calibration-script) produced, for
`Qwen3-30B-A3B-Instruct-2507` at `max_seqlen=16384, samples=24`:

```json
{
  "prefill": {"a": 26.7158,   "b":  7.0113},
  "decode":  {"a": 1.837e-5,  "b": 20.0000}
}
```

Evaluated per phase:

| target | prefill scale factor | decode scale factor |
| ------ | -------------------- | ------------------- |
| 0.3    | `2.19e+2`            | `7.41e-3`           |
| 0.5    | `8.90e+2`            | `4.05e-1`           |
| 0.7    | `3.62e+3`            | `2.21e+1`           |

So at a 70% target you'd launch vLLM with:

```bash
--attention-config.skip_softmax_threshold_scale_factor_prefill 3616.16 \
--attention-config.skip_softmax_threshold_scale_factor_decode  22.09
```

**You do not need to pick a single target sparsity up front** — the
calibration output is `(a, b)` itself, so you can evaluate the formula
at several target sparsities (0.3, 0.5, 0.7, …) and generate one vLLM
config per operating point without re-running calibration. The driver
script supports this via `--also-evaluate 0.3 0.5 0.7`.

---

## 6. Serving vLLM with the calibrated values

```bash
PREFILL=$(jq -r .vllm_flags.skip_softmax_threshold_scale_factor_prefill calibration/skip_softmax.json)
DECODE=$( jq -r .vllm_flags.skip_softmax_threshold_scale_factor_decode  calibration/skip_softmax.json)

vllm serve openai/gpt-oss-120b \
  --data-parallel-size 8 \
  --max-model-len 131072 \
  --attention-config.skip_softmax_threshold_scale_factor_prefill "$PREFILL" \
  --attention-config.skip_softmax_threshold_scale_factor_decode  "$DECODE"
```

Only backends that advertise `supports_skip_softmax()` (currently
FlashInfer, FlashInfer-MLA, and FlashInfer-MLA-sparse — see
`vllm/v1/attention/backends/*.py`) will honour the flags. On other
backends the values are silently ignored.

Pass only the side you calibrated: if you set
`target_sparse_ratio.decode = 0.0`, omit the `_decode` flag and the
decode path will run dense softmax.

---

## 7. Validating the chosen scale factors

Use the existing sweep infrastructure on this branch to confirm that
the calibrated numbers actually deliver the expected
performance/accuracy trade-off:

1. **Perf** — run `bash tests/skip_softmax_perf.sh <model> <prefill> <decode> auto 8000`
   and compare TTFT / TPOT at ISL ∈ {8192, 65536, 131072} against the
   baseline (no flags).
2. **Accuracy** — run `bash tests/skip_softmax_accuracy.sh ...` which
   covers GSM8K, MMLU-Pro, and LongBench-E. LongBench-E's 8k+ bucket
   is the most sensitive signal and usually the one that disqualifies
   an aggressive target sparsity.
3. **Sanity bound** — a scale factor of `0.0` must reproduce the
   dense-softmax baseline exactly. If calibration returns something
   tiny (e.g. `1e-9`), treat that as "no benefit at this target
   sparsity" and try a higher `target_sparse_ratio`.

For richer context on the sweep, see
[`tests/skip_softmax_test_plan.md`](skip_softmax_test_plan.md).

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `ValueError: method must be a string` / config errors | `sparse_cfg` structure wrong | Copy the template in §3 verbatim; note the top-level `"calibration"` key lives next to the layer-pattern entries, and the layer pattern should be `"*"` (not `"*attention*"`). |
| `Inserted 0 sparse attention modules` | Wildcard pattern didn't match any registered sparse module — typically `"*attention*"` against modules named `*self_attn*` | Use `"*"` or `"*self_attn*"`. See §3 callout on pattern matching. |
| `ValueError: Using a 'device_map' ... requires 'accelerate'` | `accelerate` not installed | `pip install accelerate`. |
| `ModuleNotFoundError: No module named 'nltk'` / `'wonderwords'` | RULER dataset generator deps missing | `pip install nltk wonderwords` and `nltk.download('punkt'); nltk.download('punkt_tab')`. |
| `AttributeError: 'NoneType' object has no attribute 'to'` inside `flash_attention_forward` during **decode** calibration | transformers 5.x FA integration dereferences `s_aux` unconditionally | Apply the monkey-patch in §4.1 (switches the fast prefill to SDPA). |
| `CUDA error: unspecified launch failure` raised at `torch.cuda.empty_cache()` during chunked prefill | Allocator fragmentation on long seqlens | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and rerun; see §4.2. |
| Softmax patch never fires | Model loaded with `attn_implementation="sdpa"` or `"flash_attention_2"` | Reload with `attn_implementation="eager"`. Model-Optimizer's `pytorch` backend only patches the eager path. |
| OOM during prefill calibration | `chunk_size` too large for long `max_seqlen` | Lower `chunk_size` (e.g. 1024) or `max_seqlen`. |
| Decode calibration raises `RuntimeError: calibration_data must be built` | You passed a custom `forward_loop` for prefill only | Either let the RULER builder run (leave `forward_loop=None`), or set `target_sparse_ratio.decode = 0.0` to skip decode. |
| Calibration reports "fitted b=20" or similar optimiser-bound value | Fit ran into its internal clamp — usually decode with a very steep sparsity–threshold curve | See §4.4: increase `num_decode_tokens` / `samples`, or fall back to reading the scale factor directly from the raw sweep table. |
| Target sparsity lies outside the printed per-threshold sweep (`Avg Sparsity` column never reaches it) | `max_seqlen` too short — sparsity saturates before your target | Re-calibrate at a larger `max_seqlen`. See §4.3. |
| Calibration returns `R²` far below 1 for a phase | Sparsity is a weak function of threshold at your chosen `max_seqlen`, or not enough samples | Increase `samples`, widen `threshold_trials`, or re-calibrate at a `max_seqlen` closer to your serving ISL. |
| vLLM perf unchanged after setting the flags | Backend doesn't implement skip-softmax, or the value is too small relative to `seq_k` | Check `cls.supports_skip_softmax()` for your backend; try a larger `target_sparse_ratio` and re-derive the scale factor. |

---

## 9. TL;DR

1. Install `nvidia-modelopt[torch]` **plus** `accelerate`, `nltk`
   (with `punkt`/`punkt_tab`), and `wonderwords`.
2. Load the model with `attn_implementation="eager"` and
   `device_map="auto"`.
3. Apply the decode-forward-loop monkey-patch from §4.1 if you're on
   transformers 5.x, or just run
   [`tools/calibrate_skip_softmax.py`](../tools/calibrate_skip_softmax.py)
   which bakes it in.
4. Call `mtsa.sparsify(model, cfg_with_calibration)` with your desired
   `target_sparse_ratio` per phase and a `max_seqlen` where the raw
   sparsity sweep clearly reaches (target + 10%).
5. Export with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
   avoid the async CUDA launch failure at long seqlens.
6. Read `(a, b)` off any attention module's
   `_sparse_method_instance.calibration_params`; compute
   `scale = a * exp(b * target_sparsity)` per phase.
7. Cross-check the scale factor against the printed raw sweep table
   before trusting the fit (especially for decode).
8. Pass the two numbers to
   `--attention-config.skip_softmax_threshold_scale_factor_{prefill,decode}`.
9. Validate with `tests/skip_softmax_perf.sh` +
   `tests/skip_softmax_accuracy.sh`.
