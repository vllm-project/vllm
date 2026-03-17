# Helion Kernel Versioning

Helion kernel configs are autotuned for specific kernel implementations. When a
kernel's algorithm changes (e.g., new tiling strategy, different Helion features),
old configs may produce suboptimal or incorrect results. The versioning system
lets a new implementation coexist alongside the old one while configs are being
autotuned on target platforms.

## How It Works

Each kernel registration carries a version number:

```python
@register_kernel("silu_mul_fp8")       # ver=1 (default)
def silu_mul_fp8(x, y, scale): ...

@register_kernel("silu_mul_fp8", ver=2)
def silu_mul_fp8(x, y, scale): ...     # new implementation, same signature
```

Both versions can live in the same source file (e.g., `ops/silu_mul_fp8.py`).
The function name at the module level doesn't matter — the `@register_kernel`
decorator captures the function before any shadowing occurs.

Config files are stored per version: `configs/silu_mul_fp8_v1.json`,
`configs/silu_mul_fp8_v2.json`. This is the `versioned_name` property on
`HelionKernelWrapper`.

## Version Resolution

At import time, `ops/__init__.py` resolves the best version of each kernel for
the current GPU platform:

1. Iterate versions newest-first.
2. Return the first version that has configs for the current platform.
3. If falling back to an older version, emit a `DeprecationWarning`.
4. If no version has configs, skip the kernel (logged at DEBUG level).

Callers are version-unaware — they import `silu_mul_fp8` from `ops` and get
whichever version was resolved.

## Adding a New Version

1. Add the new implementation in the same source file with `ver=N`:
   ```python
   @register_kernel("my_kernel", ver=2)
   def my_kernel(x, y): ...
   ```
2. The new version **must have the same function signature** (parameter names
   and type annotations) as the previous version, since callers are
   version-unaware.
3. Autotune the new version on target platforms:
   ```bash
   python scripts/autotune_helion_kernels.py --kernels my_kernel:2
   ```
   This writes `configs/my_kernel_v2.json`.
4. Once the new version has configs for all core platforms, it will be
   automatically selected by the resolver.

## Retiring an Old Version

Once the new version has configs for all platforms you care about, delete
the old version's `@register_kernel` function and its config JSON file.

## CI Policy Tests

`tests/kernels/helion/test_versioning.py` enforces these rules:

- **Max 2 versions**: No kernel may have more than 2 registered versions at
  once. Add configs for the new version and remove the old one before adding
  a third.
- **Core platform coverage**: When 2 versions coexist, the newest must have
  configs for all `CORE_PLATFORMS` (defined in `platforms.py`). This prevents
  merging a new version that has no configs at all.
- **Signature compatibility**: All versions of a kernel must have identical
  parameter names and type annotations.

## Autotuning

```bash
# Autotune latest version of all kernels
python scripts/autotune_helion_kernels.py

# Autotune a specific version
python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8:2

# List registered kernels and their versions
python scripts/autotune_helion_kernels.py --list

# Force re-autotune even if configs exist
python scripts/autotune_helion_kernels.py --force
```
