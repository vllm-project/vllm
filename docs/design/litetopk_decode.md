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

The MXFP4 route adds H32/D128, page128, BF16 top512 and 1..6 query tokens per
request. It requires `get_paged_mqa_logits_bf16_metadata` and
`fp4_paged_mqa_logits_bf16` from the companion DeepGEMM patch. Decode head
weights are converted to BF16; scores and ties are selected exactly in that
dtype. BF16 rounding can change both selected tokens and published candidate
blocks relative to the default FP32 path. Prefill retains its current dtype.

The CPU metadata hint `write_max_decode_len` selects Q4/three TMEM stages for
1..4 tokens or Q6/two TMEM stages for 5/6. Flattened Q has a size-one token
axis and cannot supply that hint. Each BF16 call builds its own split384
schedule from live causal lengths and request IDs, including during CUDA graph
replay. This adds schedule and cast overhead per eligible layer, included in
any complete-chain measurement; it never consumes generic split256 metadata.
Candidate mask consumers use the existing path because masking would invalidate
the producer's histogram. Candidate sources publish from the same BF16 scores
consumed by LiteTopK.

Serving route tests: `.venv/bin/python -m pytest tests/v1/attention/test_litetopk_decode.py`.
