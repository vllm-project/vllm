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

## `patch-flashinfer-sm89-mma-scales.py` + `mma_sm120.cuh`

**Required.** Both problems it fixes are silent — no crash, no warning.

Ada has no hardware block-scaled MMA, so the `+sm89` port replaces

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
```

with a plain FP8 MMA followed by a scale multiply. The 0.6.14 port gets two
things wrong:

**1. One scale for four accumulators.** In the m16n8k32 fragment layout `d0`/`d1`
belong to row `gid` and `d2`/`d3` to row `gid+8`, while adjacent columns carry
distinct B scales. `scale_vec::1X` gathers those distributed scale operands per
accumulator; the manual substitution applied the calling lane's single
`scale_a`/`scale_b` product to all four, so three of the four are scaled with
the wrong factor whenever scales differ across the fragment. This is invisible
when every block in a tile shares a scale and grows with the number of distinct
KV blocks in flight — it surfaces as **long-context quality decay**, not as an
obvious failure. Measured on 8×4090 / TP4×PP2 with a 56K-token prompt at
`temperature=0`, unique-4-gram rate over three runs went **53.0 / 69.6 / 84.8 %
before the fix → 90.0 / 90.0 / 91.9 % after**.

**2. No range handling for degenerate E8M0 encodings.** `scale_exp <= 0` — which
every all-zero or padded block hits — wraps through `uint32` to `0xFF800000`,
i.e. `-Inf`, which becomes NaN and spreads across attention, so output is
garbage from the first token. `0xff` (the E8M0 NaN encoding) and exponent
overflow are likewise unhandled.

Both are fixed upstream in `flashinfer_python-0.6.17+sm89.1`. `mma_sm120.cuh`
here is that release's revision of the file, taken **verbatim** — outside the two
hunks above it is byte for byte identical to the 0.6.14 one, so installing it is
exactly equivalent to applying them.

This is JIT source text compiled by FlashInfer at runtime, so no build toolchain
is needed — but point `FLASHINFER_CACHE_DIR` at a fresh directory afterwards, or
stale cached kernels are reused and the fix does nothing.

```bash
python assets/patch-flashinfer-sm89-mma-scales.py --check   # dry run, writes nothing
python assets/patch-flashinfer-sm89-mma-scales.py           # apply
```

The script recognises the file by md5 and refuses anything it does not know,
rather than overwriting blind:

| md5 | state | action |
|---|---|---|
| `8cff4db04e3dbb41839cf1329872a703` | 0.6.14+sm89 as released | patch |
| `beab1e4ae9921b19b261e685887f866f` | + the earlier clamp-only fix (superseded) | patch |
| `594b3362ffdd9ad4e15e92e836a2bf5f` | already fixed | no-op |
| anything else | unknown | exit non-zero, write nothing |

It leaves a `.orig` backup and re-checks the md5 after writing, so a partial or
wrong install fails loudly instead of shipping.

> An earlier revision of this repository carried a clamp-only fix, which
> addressed problem 2 but not problem 1. A deployment patched that way starts
> cleanly, answers short prompts correctly, and quietly degrades on long inputs.
> If you installed it, re-run the script above — it detects that state and
> upgrades it.
