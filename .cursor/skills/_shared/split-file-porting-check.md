# Split-File Porting Check

Use after an upstream rebase, cherry-pick, or sync when upstream **splits,
renames, or deletes** a file that previously carried Cohere customizations.
Marker checks on the old path alone are not sufficient — runtime may import
the new package path instead.

## When to run

- Upstream deleted or gutted a monolithic file and added a package directory
  (e.g. `compressed_tensors_moe.py` → `compressed_tensors_moe/`)
- Rebase conflict resolution kept custom hunks in a legacy file upstream removed
- `check-cohere-markers` passes but behavior regresses (orphaned custom code)

## Procedure

### 1) Detect upstream file splits

Between `<old-tag>` and `<new-tag>` (or `UPSTREAM_REF`..`HEAD`):

```bash
OLD=v0.19.1
NEW=v0.21.0   # confirm with user

git diff --name-status "upstream/$OLD".."upstream/$NEW" -- \
  vllm/model_executor/layers/quantization/compressed_tensors/ \
  vllm/model_executor/layers/pooler/ \
  vllm/model_executor/layers/fused_moe/ \
  | grep -E '^D|^A|^R'
```

Flag pairs where a **deleted** path (`D`) and **added** directory/package (`A`)
cover the same behavior.

**Known watch list (v0.19 → v0.21):**

| Legacy path (may still exist on fork) | Successor package / path | Upstream PR |
|---------------------------------------|--------------------------|-------------|
| `compressed_tensors/compressed_tensors_moe.py` | `compressed_tensors_moe/` | vllm#38960 |
| `layers/pooler.py` | `layers/pooler/` | upstream pooler split |
| `quantization/mxfp8.py` | `quantization/online/mxfp8.py` | online-quant frontend |

### 2) Diff Cohere hunks: old path vs new package paths

For each split pair `OLD_PATH` → `NEW_DIR/`:

```bash
OLD_PATH=vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
NEW_DIR=vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe

echo "=== Cohere hunks in legacy file (fork vs upstream new tag) ==="
git diff "upstream/$NEW" -- "$OLD_PATH" | grep -E '^\+.*cohere|^\-.*cohere|cohere start|cohere end' || true

echo "=== Cohere hunks in new package (fork vs upstream new tag) ==="
git diff "upstream/$NEW" -- "$NEW_DIR/" | grep -E '^\+.*cohere|^\-.*cohere|cohere start|cohere end' || true
```

Also list marker blocks still only in the legacy file:

```bash
rg -n 'cohere start|# cohere|// cohere' "$OLD_PATH" 2>/dev/null || true
rg -n 'cohere start|# cohere|// cohere' "$NEW_DIR/" 2>/dev/null || true
```

**Fail / stop** if legacy has Cohere hunks that do not appear in the successor
package (same behavior, not just same comment text).

### 3) Confirm runtime import path

Grep importers to see which module is actually used:

```bash
# Example: MoE dispatch
rg -n 'from vllm\.model_executor\.layers\.quantization\.compressed_tensors\.compressed_tensors_moe' \
  vllm/model_executor/layers/quantization/

# Resolve package vs legacy file
python3 -c "
import importlib.util
spec = importlib.util.find_spec(
    'vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe'
)
print('origin:', spec.origin if spec else 'NOT FOUND')
print('submodule_search_locations:', getattr(spec, 'submodule_search_locations', None))
"
```

If `submodule_search_locations` points at `compressed_tensors_moe/` (package),
the legacy sibling file `compressed_tensors_moe.py` is **not** imported unless
something imports it explicitly.

### 4) Flag duplicate implementations

Both legacy monolithic file and package must not each define the same live
behavior:

```bash
LEGACY=compressed_tensors_moe.py
PKG_DIR=compressed_tensors_moe

# Legacy file still on disk?
test -f "vllm/model_executor/layers/quantization/compressed_tensors/$LEGACY" \
  && echo "WARN: legacy $LEGACY still present"

# Same symbol in both?
for sym in get_moe_method CompressedTensorsWNA16MarlinMoEMethod _normalize_weight_actorder; do
  echo "--- $sym ---"
  rg -l "$sym" "vllm/model_executor/layers/quantization/compressed_tensors/$LEGACY" 2>/dev/null || true
  rg -l "$sym" "vllm/model_executor/layers/quantization/compressed_tensors/$PKG_DIR/" 2>/dev/null || true
done
```

| Finding | Severity | Action |
|---------|----------|--------|
| Cohere hunks only in legacy file | **Regression** | Port hunks into package successor paths |
| Both paths define `get_moe_method` (or equivalent entry) | **Orphan risk** | Confirm importers; delete or thin legacy file |
| Package has no Cohere hunks but legacy does | **Regression** | Port before merge |
| Neither path has hunks; doc says customization required | **Missing** | Re-read `models-and-inference.md` |

### 5) Report template

```markdown
## Split-file porting check

| Legacy path | Successor | Cohere hunks in legacy | Cohere hunks in successor | Runtime import |
|-------------|-----------|------------------------|---------------------------|----------------|
| `.../compressed_tensors_moe.py` | `.../compressed_tensors_moe/` | yes | no | package |

**Action required:** [port / remove legacy / update doc]
```

## Related skills

- `rebase-assistant` — Step 6b after rebase
- `review-models-inference-upstream-diff` — Step 2b during restructuring review
- `check-cohere-markers` — markers on modified upstream files (complements, does not replace, this check)
