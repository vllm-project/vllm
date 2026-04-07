Title: compat: Fix Ernie4.5 RoPE initialization under Transformers v5

What

This PR makes two targeted fixes in `vllm/transformers_utils/config.py` to ensure models like `Ernie4_5_VLMoeForConditionalGeneration` initialize correctly when running against Transformers v5:

- Remove the Transformers v5 auto-validator for RoPE (`validate_rope`) from `PretrainedConfig.__class_validators__` at module load time.
- After calling `config.standardize_rope_params()` in `patch_rope_parameters()`, propagate a real `config.rope_theta` into `config.rope_parameters['rope_theta']` when the dict contains `None`.

Why

Transformers v5 triggers RoPE validation during `PretrainedConfig.__init__()`. Some remote configs (e.g., Ernie-4.5) set `rope_theta` only after `super().__init__()`, causing the auto-validation to see a missing/None `rope_theta` and throw a KeyError. vLLM already performs proper RoPE patching and validation in `patch_rope_parameters()`; deferring/removing the premature auto-check removes false failures while preserving validation coverage.

How

- `_disable_rope_auto_validation()` removes `validate_rope` from the class validators for Transformers >= 5 at module import.
- `patch_rope_parameters()` now fills `rope_parameters['rope_theta']` from `config.rope_theta` when necessary before calling `config.validate_rope()`.

Testing Done

- Verified `transformers` v5.5.0 installed in `.venv` and that `get_config('baidu/ERNIE-4.5-VL-28B-A3B-PT', trust_remote_code=True, ..)` loads without KeyError; `rope_theta` is present and set to the expected default (500000).
- Verified `AutoConfig.from_pretrained()` (tokenizer codepath) also succeeds.
- Ran `pre-commit` hooks; `ruff` auto-format applied and all hooks passed.

Notes / Caveats

- The module-level suppression mutates Transformers' `__class_validators__` to remove `validate_rope`. This is intentionally narrow (only removes the RoPE auto-validator) and vLLM still calls `validate_rope()` explicitly after patching, preserving validation coverage.
- On macOS, running the full test suite locally can show unrelated fork/Objective-C initialization errors (SIGSEGV). Recommend running full regression on Linux CI.

How to verify locally

```bash
# Ensure venv activated and transformers v5 installed
.venv/bin/python -c "import transformers; print(transformers.__version__)"
# Quick config load test
.venv/bin/python -c "from vllm.transformers_utils.config import get_config; c=get_config('baidu/ERNIE-4.5-VL-28B-A3B-PT', trust_remote_code=True); print(getattr(c.get_text_config(),'rope_theta', None), c.get_text_config().rope_parameters.get('rope_theta'))"
# Run the targeted pytest (macOS: set fork safety env)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
.venv/bin/python -m pytest 'tests/models/test_initialization.py::test_can_initialize_large_subset[Ernie4_5_VLMoeForConditionalGeneration]' -q
```

Suggested reviewers / labels

- Reviewers: @maintainer-rope, @transformers-compat
- Labels: compat, bug, tests

If preferred, we can instead take a smaller-scope approach (context-managed suppression) but that proved fragile because multiple codepaths call `AutoConfig.from_pretrained()` (tokenizer path, etc.).

Co-authored-by: GitHub Copilot <>
