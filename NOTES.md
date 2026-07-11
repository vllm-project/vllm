# Issue #48238 — Gemma4 dense NVFP4 fails in `quant_method.tie_weights`

**Headline: the crash is already fixed on `main`, and the fix is present in this tree.**
The bug is a release-lag problem, not a live `main` defect. The only real gap left in
`main` is that the fix landed with no regression test, which is what this change adds.

## 1. Root cause

`nvidia/Gemma-4-31B-IT-NVFP4` has `tie_word_embeddings: true`, so
`Gemma4ForCausalLM.__init__` runs `self.lm_head.tie_weights(self.model.embed_tokens)`
(`vllm/model_executor/models/gemma4.py:1561`). `ParallelLMHead.tie_weights` delegates to
the layer's quantization method (`vocab_parallel_embedding.py:555`).

Before PR #42124, ModelOpt's `get_quant_method` ignored `ParallelLMHead`, so a quantized
model's `lm_head` kept `UnquantizedEmbeddingMethod`, which implements `tie_weights`
(`vocab_parallel_embedding.py:80`). #42124 added `ParallelLMHead` to ModelOpt's dispatch
so the head could be quantized; as a side effect, an *excluded* head now gets
`UnquantizedLinearMethod` (`modelopt.py:192-193`) instead of `UnquantizedEmbeddingMethod`.
That class inherits `tie_weights` from `QuantizeMethodBase`, which at the time did nothing
but `raise NotImplementedError` — hence the reported traceback, at init, before any weight
loading.

Worth noting because the issue text gets it slightly wrong: this checkpoint's
`hf_quant_config.json` *does* list `lm_head` in `exclude_modules` (I read it from the Hub),
so the head was never actually NVFP4-quantized. The failing method was
`UnquantizedLinearMethod`, and the missing implementation was in the **base class**, not in
a ModelOpt-specific one.

## 2. The fix, and why there is no source change here

Upstream PR #45544 ("Default tie_weights to sharing the weight", merged 2026-06-26,
`ad28d60`) gave `QuantizeMethodBase.tie_weights` a default that shares the tensor
(`layer.weight = embed_tokens.weight`) — exactly what `ParallelLMHead.tie_weights` used to
do inline. That code is already in this checkout at
`vllm/model_executor/layers/quantization/base_config.py:53-65`, so on `main` this
checkpoint takes: excluded `lm_head` → `UnquantizedLinearMethod` → default `tie_weights`
→ shared embedding weight → no crash.

The reporter is on the v0.24.0 wheel, whose release branch forked before `ad28d60` landed;
a backport request is already open on the PR (and a backport is not something that can be
done from a `main` working tree). Writing a second fix for an already-fixed bug would be
duplicate work, which `AGENTS.md` tells me to refuse, so **I deliberately changed no
production code.**

What `main` genuinely lacks is coverage: no test in `tests/` referenced `tie_weights` at
all before this change (`grep -rn tie_weights tests/` → nothing), which is precisely why
#42124 could regress it silently. I added one focused regression test that pins the
contract along the exact call chain from the traceback:

`tests/quantization/test_modelopt.py::test_modelopt_excluded_lm_head_ties_to_embedding_weight`

It builds an NVFP4 config shaped like the real checkpoint (`exclude_modules=["lm_head"]`,
`kv_cache_quant_algo="FP8"`), asks it for the quant method of a `ParallelLMHead` at prefix
`language_model.lm_head` (the prefix `Gemma4ForConditionalGeneration` uses, which also
exercises ModelOpt's substring exclusion), asserts the method is `UnquantizedLinearMethod`,
then calls `tie_weights` and asserts the head's weight *is* the embedding's tensor. It
fails with `NotImplementedError` against pre-#45544 code and passes now.

The test file already covers ModelOpt lm_head quant-method *selection*
(`test_modelopt_nvfp4_quantizes_parallel_lm_head` and friends), so this extends that suite
rather than adding a file, and reuses its `_mock_lm_head` helper.

## 3. Files changed

- `tests/quantization/test_modelopt.py` — added the regression test plus a small
  `_weight_holder()` helper (a bare `nn.Module` with a `weight` parameter) next to the
  existing `_mock_lm_head()`.
- `NOTES.md` — this file.

## 4. Risk / uncertainty

- **Test-only change; zero runtime risk.** No production behavior is touched.
- **Residual hazard I chose not to "fix".** The #45544 default shares the embedding's
  bf16 tensor into *whatever* head exists, including one whose `create_weights` allocated
  packed-FP4 params. If a ModelOpt checkpoint ever tied embeddings *without* listing
  `lm_head` in `exclude_modules`, `main` would attach `ModelOptNvFp4LinearMethod`, silently
  overwrite its uint8 weight with a bf16 tensor, and then run an FP4 GEMM over it — garbage
  logits instead of a crash. I checked three real tied ModelOpt checkpoints
  (`Gemma-4-31B-IT-NVFP4`, `Gemma-4-26B-A4B-NVFP4`, `diffusiongemma-26B-A4B-it-NVFP4`) and
  all three exclude `lm_head`, so this is not reachable today. Guarding it (e.g. forcing an
  unquantized method when `tie_word_embeddings` is set) would be a speculative behavior
  change beyond this issue's scope, so I left it alone and am flagging it instead.
- The test deliberately does not assert the tied-*and*-quantized case, to avoid enshrining
  the behavior described above as correct.

## 5. How I verified

- **I did not run pytest.** This environment has no vLLM build (`import torch` →
  `ModuleNotFoundError`, no `.venv`), so the test is unrun. It is CPU-only and needs no GPU
  or distributed init — it constructs no real `ParallelLMHead` — so it should run anywhere
  the rest of `tests/quantization/test_modelopt.py` runs:
  `pytest tests/quantization/test_modelopt.py -k tie -v`. That is the one claim in these
  notes I could not confirm by execution.
- `ruff check` and `ruff format --diff` on the changed file: clean; `python3 -m py_compile`:
  clean; longest new line is 85 chars (limit 88).
- Traced the dispatch by hand against the checkpoint's real `hf_quant_config.json`
  (fetched from the Hub): `is_layer_excluded("language_model.lm_head")` matches the
  `"lm_head"` entry via ModelOpt's substring rule (`modelopt.py:165-174`), so
  `get_quant_method` returns `UnquantizedLinearMethod`, whose `tie_weights` resolves to
  `QuantizeMethodBase.tie_weights`.
- Duplicate-work check (per `AGENTS.md`; done over the web, as `gh` was unavailable): no
  open PR references #48238, and the only open PR touching tied lm_head weights is #45535,
  which is compressed-tensors WNA16, a different backend.
