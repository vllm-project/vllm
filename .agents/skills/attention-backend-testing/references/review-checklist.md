# Review checklist for PRs touching an attention backend

Use alongside the `vllm-pr-review` skill. Ask only for what the PR's change
can actually break.

## Scope the change

- Which rows of the backend's dispatch table does the diff touch? If the diff
  adds a branch, the PR must add a test for it that spies on the kernel.
- Does it change a capability declaration (head sizes, KV dtypes, block sizes,
  sinks, SWA, PCP/DCP, CUDA graph support level, KV layouts)? Then the
  selector accept/reject tests must change with it.
- Does it change the builder? Check for new `.item()`, `.cpu()`, `.tolist()` on
  the decode-only path, new per-build allocations, and buffers that are
  allocated for configs that never read them.
- Does it plan prefill or context chunks per request? A chunk planner must
  never emit a chunk with no query tokens: requests with query length 0 sit in
  the prefill region after FULL CUDA graph padding or an async KV connector
  load (vllm-project/vllm#55695, #51489).
- Does it change an op or bump the kernel library? Every backend that calls the
  op is affected; run their suites, not only the one in the title.

## Evidence to ask for

| Change | Tests | Perf | Accuracy |
| --- | --- | --- | --- |
| New dispatch branch or kernel | L1 op + L2 dispatch (spy) + correctness rows for the new condition | benchmark config keys covering the condition, compared to merge base | GSM8K for one model that takes the branch |
| Builder / metadata change | L2 builder + one L3 run per CUDA graph mode | if decode path changed | if logits can change |
| Capability change | L0 contract both directions | no | if it changes which backend real models select |
| Default/priority change | L0 + L3 on affected models | yes | yes |
| Refactor with no intended behavior change | existing suite passes on the right hardware | no, unless hot path | no |

## Coverage that is easy to miss

- Shuffle/alternate KV layouts enabled by env.
- fp8 KV with non-unit scales.
- Spec decode (multi-token decode, non-uniform q lengths).
- Sliding window with context longer than the window in extend and decode.
- KV sharing / KV-update-bypass layers.
- Full CUDA graph replay with changed block tables.
- Requests with query length 0 after the real prefills (graph padding, async
  KV connector loads), combined with prefill chunking.
- Model runner v2 and async scheduling.
- The test file is actually collected by a Buildkite step that runs on the
  target GPU, and the backend file is in that step's
  `source_file_dependencies`.
