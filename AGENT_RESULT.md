## Root Cause

`EPLBConfig.num_redundant_experts` defaulted to `0`, giving users no way to distinguish "I set this to 0" from "I never set this." When EPLB is first enabled on a model where `num_routed_experts % ep_size != 0`, the user had to manually compute and set the minimum valid value or hit a `ValueError` in `FusedMoE.__init__`. The minimum valid value is deterministic from the startup config, so the default should compute it automatically.

## Change Made

**`vllm.config.parallel.EPLBConfig`**
- Changed `num_redundant_experts: int = Field(default=0, ge=0)` to `num_redundant_experts: int | None = Field(default=None, ge=0)` - `None` means "auto-compute minimum."
- Added `EPLBConfig.get_num_redundant_experts(num_routed_experts: int, ep_size: int) -> int` method that returns `(-num_routed_experts) % ep_size` when the field is `None` (the smallest r >= 0 such that the total is divisible by ep_size), and the explicit value otherwise.

**`vllm.config.parallel.ParallelConfig._validate_parallel`**
- Fixed the "EPLB not enabled but num_redundant_experts is set" check to treat `None` the same as not-set, avoiding a false error on the new default.

**All MoE layer classes** (22 model files + `transformers/moe.py`)
- Replaced every direct `eplb_config.num_redundant_experts` read with a call to `eplb_config.get_num_redundant_experts(n_routed_experts, ep_size)` guarded by `enable_eplb`, so the auto-minimum is used when the field is `None` and 0 is used when EPLB is off.
- Removed the redundant `None`-to-`0` normalization blocks in `exaone_moe` and `laguna` that pre-dated this change.

Touched model modules: `afmoe`, `AXK1`, `deepseek_v2`, `ernie45_moe`, `exaone_moe`, `glm4_moe`, `hunyuan_v1`, `hy_v3`, `interns1_pro`, `laguna`, `lfm2_moe`, `llama4`, `mimo_v2`, `mixtral`, `nemotron_h`, `nemotron_h_mtp`, `openpangu`, `qwen3_5`, `qwen3_moe`, `qwen3_next`, `step3p5`, `transformers.moe`.

## Testing

- `pytest --noconftest tests/distributed/test_eplb_utils.py`: **5 passed**
- Manual smoke test of `EPLBConfig.get_num_redundant_experts`: verified auto-minimum logic (e.g., 256 experts / ep_size 7 -> r=3 since 259=7*37) and explicit value pass-through.
- Confirmed no remaining direct `eplb_config.num_redundant_experts` reads in model files.

## Lint

- `python -m ruff check` on all changed files: **All checks passed** (the one error reported is in `lfm2_moe.py` line 712 and is pre-existing, unrelated to this change).

