1. **Fix `ignore_file_pattern` in Python code**
   - Replace `ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"]` with `ignore_file_pattern=["*.pt", "*.safetensors", "*.bin"]` in `benchmarks/backend_request_func.py` and `vllm/tokenizers/registry.py`.
   - The current `.*.pt` patterns are technically incorrect glob patterns and were only matching dotfiles, which defeats the purpose of ignoring weights. The `*.pt` glob patterns are correctly supported in `modelscope` versions other than v1.15.0 and correctly ignore model weights.

2. **Exclude `modelscope!=1.15.0` in package dependencies and scripts**
   - In `docker/Dockerfile`, replace `'modelscope<1.38'` with `'modelscope<1.38,!=1.15.0'`
   - In `.buildkite/test-amd.yaml`, replace `'modelscope<1.38'` with `'modelscope<1.38,!=1.15.0'`
   - In `.buildkite/test_areas/misc.yaml`, replace `'modelscope<1.38'` with `'modelscope<1.38,!=1.15.0'`
   - In `requirements/test/xpu.in`, replace `modelscope<1.38` with `modelscope<1.38,!=1.15.0`

3. **Complete pre-commit steps**
   - Use `pre_commit_instructions` tool to make sure proper testing, verifications, reviews and reflections are done.

4. **Submit changes**
   - Push and verify cleanly.
