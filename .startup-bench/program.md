# Autonomous startup-optimization loop

You are iterating on vLLM's cold-start time via the autoresearch pattern. The
framework contract is in `CLAUDE.local.md` § "Startup optimization experiments".
This file is the loop you run every iteration. Keep it loaded.

## In-scope files (read before starting)

- `vllm/entrypoints/openai/api_server.py` — CLI entrypoint
- `vllm/__init__.py` — top-level eager imports (often the biggest single target)
- `vllm/v1/engine/llm_engine.py` (and/or `vllm/engine/llm_engine.py`)
- `vllm/config.py`, `vllm/engine/arg_utils.py`
- Any file the baseline `import-time` profile fingers

Run once to rank top offenders:

```bash
cd /tmp && .venv/bin/python -X importtime -c \
  "import vllm.entrypoints.openai.api_server" 2> /tmp/importtime.log
# inspect /tmp/importtime.log — fields are cumulative, self, and module name
```

## The loop

LOOP FOREVER (until interrupted or a clear budget limit — see § "When to stop"):

1. **Inspect state.** `git status` (should be clean) and `git log --oneline -5`.
2. **Form ONE hypothesis** with an expected wall-clock impact. One sentence.
   Example: "Defer `import transformers` out of `vllm/__init__.py`; expect ~2s
   cold win because baseline importtime shows it's ~2.1s self."
3. **Edit code.** Be surgical. Small diffs. Don't mix hypotheses.
4. **Commit:**
   ```bash
   git add -A && git commit -m "[startup] <one-line hypothesis>"
   ```
5. **Measure:**
   ```bash
   SHA=$(git rev-parse --short HEAD)
   .venv/bin/python .startup-bench/measure.py --tag apr16 --config qwen-0.5b \
     > .startup-bench/logs/apr16/$SHA.log 2>&1
   ```
6. **Read results:**
   ```bash
   awk '/^---$/{p=!p; next} p' .startup-bench/logs/apr16/$SHA.log
   ```
   If the summary block is empty or missing, the harness itself crashed:
   `tail -n 80 .startup-bench/logs/apr16/$SHA.log` and diagnose.
7. **Append to results.tsv** (tab-separated, one line):
   ```
   <sha>	<t_cold_median_s>	<t_warm_median_s>	<peak_vram_gb>	<status>	<description>
   ```
   For `crash` / `timeout` / `noisy` numeric failures: use `0.000` for missing
   numeric fields. Never `git add` results.tsv — it is gitignored by design.
8. **Decide** (compare to the most recent `keep` / `baseline` row):
   - **keep** if: `status == ok` AND `t_cold` improves by ≥ 1.0 s AND
     `t_warm` does not regress by > 0.5 s AND `correctness == pass`.
     → branch already advanced; move to next iteration.
   - **discard** otherwise. `git reset --hard <prev_kept_sha>`.
   - **crash**: bug/OOM. If it's a trivial typo in your own edit, fix and re-run
     under the same commit. If the idea is fundamentally broken, discard.
   - **noisy**: stdev threshold tripped. Wait 60 s (`sleep 60`) for the neighbor
     to clear, re-run once. Still noisy? Park, move on.
9. Go to 1.

## Keep/discard threshold rationale

`1.0 s` is set so we don't chase single-sample noise; typical shared-box noise
is a few hundred ms. If a series of 0.5 s wins are available, stack them and
measure the aggregate — commit the stack as one experiment.

## When to stop

Once the loop has begun, do **NOT** ask the human "should I continue?". The
human may be asleep. Stop only on:

- explicit user interrupt
- 30 total experiments attempted, AND no `keep` in the last 10
- experiment_timeout (30 min) tripped on two consecutive attempts

On stop, write a brief `.startup-bench/logs/<tag>/_session_summary.md`:
what you tried, what worked, what didn't, best `t_cold` achieved, next
hypotheses to try.

## Recording the description column well

Future-you reads this. Be specific:

Bad: `lazy imports`
Good: `lazy transformers import via __getattr__ in vllm/__init__.py (top-3 importtime)`

Bad: `cache stuff`
Good: `pickle resolved ModelConfig to ~/.cache/vllm-fastboot/model_cfg/<hash>.pkl (warm hit)`

## Hypothesis starters (don't treat as a todo list)

- Rank top 10 modules by `-X importtime`; for each, check: is it used during
  `--help` / startup? If not, lazy-load behind `__getattr__` or local imports.
- Fastboot cache directory layout: start with
  `~/.cache/vllm-fastboot/<vllm_version>/<gpu_arch>/<model_id_slug>/...`.
  Version key so stale caches self-invalidate on upgrade.
- First chat completion pays: (a) torch.compile graph, (b) triton autotune,
  (c) CUDA graph capture. Which dominates? Measure by toggling each.
- Model shard IO vs compile: can they run in parallel threads?
- For a 0.5B model the fraction of t_ready that is "vLLM overhead" vs
  "actual model loading" is worth knowing before optimizing either.

## Rules you cannot break

From CLAUDE.local.md (restating for the loop):

- Do not modify `.startup-bench/measure.py`.
- Do not modify `csrc/` or `CMakeLists.txt`.
- Do not add dependencies outside `requirements/*.txt`.
- Do not ship precomputed artifacts users must fetch separately.
- Do not break correctness (non-empty valid chat completion).
- Do not silently disable user-visible features (prefix cache, CUDA graphs,
  quantization) without noting it in the description.

If a hypothesis requires breaking one of these, note it in the description and
route it back to the human via a `status=blocked` row. Do not proceed.
