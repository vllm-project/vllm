# Vendored binary artifacts

## `flashinfer_python-0.6.17+sm89.1-py3-none-any.whl`

The FlashInfer build that provides the sparse-MLA kernels this fork routes to on
SM89 / Ada. Installed, it reports version `0.6.17+sm89.1`, which is what
`has_flashinfer_sparse_mla_sm89()` probes for.

| | |
|---|---|
| sha256 | `d3f483f4cf111c7f7990357112022fdeada887c2af61ec3319b9aa84930b4326` |
| size | 5840781 bytes |
| License | Apache-2.0 (`LICENSE` is bundled inside the wheel) |
| Upstream project | <https://github.com/flashinfer-ai/flashinfer> |
| Origin of this build | <https://github.com/yhfgyyf/vllm-deepseek-v4-sm89>, release `v0.23.1rc1.dev904-g998fd644b-cu132-sm89` |

Redistributed **unmodified** under Apache-2.0, which permits redistribution with
the license and notices intact; the wheel carries its own `LICENSE` file.

**Why it is vendored here.** The `+sm89` port was published as a binary only —
no source branch was ever pushed. It exists in exactly one public place, a
third-party release. If that release disappears, this configuration cannot be
rebuilt. Keeping a copy in git history removes that single point of failure.

**Why 0.6.17+sm89.1 and not 0.6.14+sm89.** The 0.6.14 port emulates the
block-scaled MMA incorrectly, in two ways that are both silent:

1. **One scale for four accumulators.** Ada has no hardware block-scaled MMA, so
   the port replaces `mma.sync...block_scale.scale_vec::1X.m16n8k32` with a plain
   FP8 MMA plus a scale multiply. In the m16n8k32 fragment layout `d0`/`d1`
   belong to row `gid` and `d2`/`d3` to row `gid+8`, while adjacent columns carry
   distinct B scales; `scale_vec::1X` gathers those distributed scale operands per
   accumulator. The 0.6.14 substitution applied the calling lane's single
   `scale_a`/`scale_b` product to all four, so three of the four are scaled with
   the wrong factor whenever scales differ across the fragment. That is invisible
   when every block in a tile shares a scale and grows with the number of distinct
   KV blocks in flight, so it surfaces as **long-context quality decay**, not as
   an obvious failure.

2. **No range handling for degenerate E8M0 encodings.** `scale_exp <= 0` — which
   every all-zero or padded block hits — wraps through `uint32` to `0xFF800000`,
   i.e. `-Inf`, which becomes NaN and spreads across attention, so output is
   garbage from the first token. `0xff` (the E8M0 NaN encoding) and exponent
   overflow are likewise unhandled.

0.6.17+sm89.1 fixes both: it rebuilds the per-accumulator scales with warp
shuffles and adds a `multiply_ue8m0()` helper that handles the NaN encoding,
overflow and subnormals.

Measured on 8×4090 / TP4×PP2 / prefix caching on, 56K-token prompt at
`temperature=0`, unique-4-gram rate over three runs — higher is better, a
collapsing value means the output has degenerated into repetition:

| FlashInfer | three runs | spread | floor |
|---|---|---|---|
| `0.6.14+sm89` | 60.7 / 81.0 / 54.4 % | 26.6 | 54.4 |
| **`0.6.17+sm89.1`** | **86.6 / 88.7 / 88.4 %** | **2.2** | **86.6** |

A second independent round on 0.6.17+sm89.1 gave 89.1 / 90.8 / 89.0 %. Short
prompts were already healthy on both. The chat quality gate passes 3/3 on all
four cases including Japanese.

### Verify you have the right file

Stock FlashInfer and the `+sm89` builds differ only in a handful of files, and
their filenames differ by one suffix, so a size check will not tell them apart.
These three probes will. Run them on any FlashInfer you are handed:

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile("flashinfer_python-0.6.17+sm89.1-py3-none-any.whl")
g = lambda n: z.read(n).decode("utf8", "ignore")
print("version  :", [l for l in g("flashinfer/_build_meta.py").splitlines()
                     if "__version__" in l])
print("sm89 gate:", g("flashinfer/mla/_core.py").count("== (8, 9)"), "(expect 2)")
print("per-acc  :", g("flashinfer/data/include/flashinfer/attention/"
                      "sparse_mla_sm120/arch/mma_sm120.cuh").count("__shfl_sync"),
      "(expect 4)")
PY
```

| probe | stock | `+sm89` 0.6.14 | `+sm89.1` 0.6.17 |
|---|---|---|---|
| `flashinfer/_build_meta.py` | `0.6.x` | `0.6.14+sm89` | `0.6.17+sm89.1` |
| `== (8, 9)` in `mla/_core.py` | 0 | 2 | 2 |
| `__shfl_sync` in `arch/mma_sm120.cuh` | 0 | 0 | **4** |

A stock build installs and imports cleanly, then rejects SM89 at backend
selection, because the sparse-MLA gate in `_core.py` never matches `(8, 9)`.

### After installing

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

Point `FLASHINFER_CACHE_DIR` at a **fresh** directory whenever you change
FlashInfer version. These kernels are JIT source text compiled at runtime; a
cache directory populated by a previous build silently keeps serving the old
kernels, so an upgrade appears to do nothing.

Check the installed version with
`python -c "import flashinfer; print(flashinfer.__version__)"`, **not `pip list`**
— pip reads its own registration metadata, which can disagree with the files
actually on disk.

Note that the vLLM wheel this fork builds against declares
`Requires-Dist: flashinfer-python==0.6.16.post3`. That pin is not satisfied by
this build, so any later `pip install` of the vLLM wheel or of a requirements
file may replace FlashInfer with the stock 0.6.16.post3 and undo the setup —
without an error. Copy `site-packages/flashinfer/` aside before running pip.
