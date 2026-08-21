# Vendored binary artifacts

## `flashinfer_python-0.6.14-py3-none-any.whl`

The FlashInfer build that provides the sparse-MLA kernels this fork routes to on
SM89 / Ada. Installed, it reports version `0.6.14+sm89`.

| | |
|---|---|
| sha256 | `d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df` |
| size | 14574383 bytes |
| License | Apache-2.0 (`LICENSE` is bundled inside the wheel) |
| Upstream project | <https://github.com/flashinfer-ai/flashinfer> |
| Origin of this build | <https://github.com/yhfgyyf/vllm-deepseek-v4-sm89> (release asset) |

**Why it is vendored here.** The `+sm89` port was published as a binary only —
no source branch was ever pushed. It exists in exactly one public place, a
third-party release. If that release disappears, this configuration cannot be
rebuilt. Keeping a copy in git history removes that single point of failure.

Redistributed unmodified under Apache-2.0, which permits redistribution with the
license and notices intact; the wheel carries its own `LICENSE` file.
