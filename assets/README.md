# Vendored binary artifacts

## `flashinfer_python-0.6.14+sm89-py3-none-any.whl`

The FlashInfer build that provides the sparse-MLA kernels this fork routes to on
SM89 / Ada. Installed, it reports version `0.6.14+sm89`, which is what
`has_flashinfer_sparse_mla_sm89()` probes for.

| | |
|---|---|
| sha256 | `95ea827b9a6303fc974f7b2872befb23efed9a3eb85074b262261e3c3944730b` |
| size | 14580531 bytes |
| License | Apache-2.0 (`LICENSE` is bundled inside the wheel) |
| Upstream project | <https://github.com/flashinfer-ai/flashinfer> |
| Origin of this build | <https://github.com/yhfgyyf/vllm-deepseek-v4-sm89>, release `v0.23.1rc1.dev1018-g8aba6ae7e-cu130-sm89` |

Redistributed **unmodified** under Apache-2.0, which permits redistribution with
the license and notices intact; the wheel carries its own `LICENSE` file. The one
change this fork needs on top of it is applied at install time by the patch
script below, never baked into the vendored file.

**Why it is vendored here.** The `+sm89` port was published as a binary only —
no source branch was ever pushed. It exists in exactly one public place, a
third-party release. If that release disappears, this configuration cannot be
rebuilt. Keeping a copy in git history removes that single point of failure.

### Verify you have the right file

Stock FlashInfer `0.6.14` and this `+sm89` build differ by about 6 KB, and their
filenames differ by one suffix — a size check will not tell them apart. These
three probes will. Run them on any FlashInfer you are handed:

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile("flashinfer_python-0.6.14+sm89-py3-none-any.whl")
g = lambda n: z.read(n).decode("utf8", "ignore")
print("version  :", [l for l in g("flashinfer/_build_meta.py").splitlines()
                     if "__version__" in l])
print("sm89 gate:", g("flashinfer/mla/_core.py").count("== (8, 9)"), "(expect 2)")
print("emulated :", "scale_exp" in g("flashinfer/data/include/flashinfer/attention/"
                                     "sparse_mla_sm120/arch/mma_sm120.cuh"),
      "(expect True)")
PY
```

| probe | stock `0.6.14` | this `+sm89` build |
|---|---|---|
| `flashinfer/_build_meta.py` | `__version__ = "0.6.14"` | `__version__ = "0.6.14+sm89"` |
| `== (8, 9)` in `flashinfer/mla/_core.py` | 0 | 2 |
| `scale_exp` in `arch/mma_sm120.cuh` | absent — uses the hardware `block_scale` MMA | present — emulates it |

A stock build installs and imports cleanly, then rejects SM89 at backend
selection, because the sparse-MLA gate in `_core.py` never matches `(8, 9)`.

## `patch-flashinfer-sm89-scale-clamp.py`

**Required.** Without it the server starts, answers, and produces garbage from
the first token — no crash, no warning.

Ada has no hardware block-scaled MMA, so the `+sm89` port replaces the
`block_scale` MMA instruction with "plain FP8 MMA, then multiply by the scale".
The exponent arithmetic is right; what is missing is the range clamp the hardware
applies to degenerate E8M0 encodings. `scale_exp <= 0` — which every all-zero or
padded block hits — wraps through `uint32` to `0xFF800000`, i.e. `-Inf`. One
`-Inf` in the accumulator becomes NaN and spreads across attention.

The fix is six lines. It is JIT source text shipped inside the wheel and compiled
by FlashInfer at runtime, so applying it needs no build toolchain — but point
`FLASHINFER_CACHE_DIR` at a fresh directory afterwards so stale cached kernels
are not reused.

```bash
python assets/patch-flashinfer-sm89-scale-clamp.py --check   # dry run, writes nothing
python assets/patch-flashinfer-sm89-scale-clamp.py           # apply
```

The script keys idempotency off a `[SM89_SCALE_CLAMP]` sentinel rather than the
anchor text, validates before it writes, and leaves a `.orig` backup. If the
anchor is missing or ambiguous it exits non-zero and touches nothing.

Applying it to a pristine copy of the vendored wheel reproduces, byte for byte,
the `mma_sm120.cuh` running in the configuration these docs describe
(md5 `beab1e4ae9921b19b261e685887f866f`).
