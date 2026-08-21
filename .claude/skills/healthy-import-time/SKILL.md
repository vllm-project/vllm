---
name: healthy-import-time
description: Find why a Python command loads slow modules. Remove imports that startup does not need, and prove that behavior stays the same. Use for slow CLI startup or import-time regressions.
---

# Find Slow Python Imports

## Diagnose

Measure the slow command before changing code.
Use `python -X importtime` to find the imports that dominate startup.
Trace each slow import to the first caller that does not need it during startup.

## Choose the fix

- Move a runtime-only import into its function.
- Put a type-only import in a `TYPE_CHECKING` block.
- Put constants and names in a small metadata module.
- Read only the config fields that the caller requests.
- Delay hardware detection until runtime needs the hardware.

Example:

```python
def run_feature() -> None:
    import large_package

    large_package.run()
```

Do not move an import that must register a handler or define a base class.

## Prove the fix

Repeat the profile after each change.

- Confirm that the slow module is absent or cheaper.
- Confirm that output and exit status do not change.
- Run the feature that uses the delayed import.
- Test help, valid input, and invalid input for a CLI.

Add a focused regression test when startup must not import a specific package.

## Measure the result

Use one unmeasured warm run. Use seven measured runs for an automated test.
Use the median as the pass condition.

Use more runs for a benchmark report:

```bash
result=$(mktemp /tmp/startup.XXXXXX.json)
command='.venv/bin/python -m package.module serve --help >/dev/null'
hyperfine --runs 11 --export-json "$result" \
  -n cold "find . -type d -name __pycache__ -prune -exec rm -rf {} + && $command" \
  -n warm "$command"
```

Hyperfine prints the mean. Read each `times` list in the JSON file. Sort the
list and use its middle value as the median.
