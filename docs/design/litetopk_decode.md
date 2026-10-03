# LiteTopK decode

Enable the experimental producer-assisted path with
`--kernel-config '{"enable_litetopk_decode": true}'`.
Build vLLM's CUDA extension and a DeepGEMM companion with
`paged_mqa_logits_histogram_version >= 1`. An external DeepGEMM installation
is preferred; source builds can also set `DEEPGEMM_SRC_DIR` to the patched
DeepGEMM checkout. An unpatched producer falls back to the existing path.

The FP8 route supports SM100, H32/D128, page64, FP32 top2048, and 1..4
query tokens per request. Prefill, unsupported shapes, context parallelism,
explicit `sparse_indexer_topk_backend` choices, and candidate mask consumers
retain the existing dispatch. Candidate sources can publish blocks from the
unmodified producer scores before selection.

DeepGEMM counts live scores into a coarse histogram while computing logits.
The CUDA selector refines the crossing bin and writes request-local logical
indices. Equal ordered score keys prefer lower logical indices; output order
is unspecified and short rows are padded with -1. This intentionally adapts
SGLang's physical-slot output and tie rule to vLLM's logical-index contract.
Candidate overflow or inconsistent counts use an exact whole-row fallback.

Each workspace lane and CUDA stream owns fixed persistent storage, shared
across layers: 1024 int32 histogram bins, 16 state bytes and 8192 eight-byte
candidates per maximum token row (about 68 KiB). Profile/warmup reserves this
storage before the workspace manager locks. Row-count changes do not move the
candidate region. The selector resets counts/candidates and balances state
counters for the next layer or CUDA graph replay.

Correctness tests: `.venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k litetopk`.
