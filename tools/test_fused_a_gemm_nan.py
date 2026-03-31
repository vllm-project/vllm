"""Test whether dsv3_fused_a_gemm contaminates clean tokens
when other tokens in the batch have NaN input.

Run on a GPU with SM >= 90 (Hopper/Blackwell):
    python tools/test_fused_a_gemm_nan.py
"""
import torch
import vllm._C  # noqa: F401  — loads the _C extension
_gemm = torch.ops._C.dsv3_fused_a_gemm


def test_nan_contamination():
    torch.manual_seed(42)
    device = "cuda"

    # DeepSeek V3 fused A projection dimensions
    hidden_size = 7168
    output_size = 2112  # q_lora_rank(1536) + kv_lora_rank(512) + rope(64)

    # bf16 weight matrix [output_size, hidden_size] stored column-major
    weight = torch.randn(output_size, hidden_size, dtype=torch.bfloat16,
                         device=device)

    for batch_size in [1, 2, 4, 7, 8, 16]:
        for nan_pattern in ["none", "all_but_first", "all_but_last", "even"]:
            # Clean input
            hidden = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16,
                                 device=device)

            # Reference output from clean input
            ref_output = torch.empty(batch_size, output_size,
                                     dtype=torch.bfloat16, device=device)
            _gemm(ref_output, hidden, weight.T)

            # Now poison some tokens with NaN
            poisoned = hidden.clone()
            if nan_pattern == "none":
                pass  # control: no NaN
            elif nan_pattern == "all_but_first":
                if batch_size > 1:
                    poisoned[1:] = float("nan")
            elif nan_pattern == "all_but_last":
                if batch_size > 1:
                    poisoned[:-1] = float("nan")
            elif nan_pattern == "even":
                poisoned[::2] = float("nan")

            # Run GEMM with poisoned input
            test_output = torch.empty(batch_size, output_size,
                                      dtype=torch.bfloat16, device=device)
            _gemm(test_output, poisoned, weight.T)

            # Check: clean tokens should produce the same output
            for tok in range(batch_size):
                is_poisoned = poisoned[tok].isnan().any().item()
                out_has_nan = test_output[tok].isnan().any().item()

                if is_poisoned:
                    # Poisoned input should produce NaN output
                    assert out_has_nan, (
                        f"batch={batch_size} pattern={nan_pattern} tok={tok}: "
                        f"NaN input but clean output!"
                    )
                else:
                    # Clean input should produce clean output
                    if out_has_nan:
                        print(f"FAIL batch={batch_size} pattern={nan_pattern} "
                              f"tok={tok}: clean input but NaN output! "
                              f"CROSS-CONTAMINATION DETECTED")
                        # Compare with reference
                        max_diff = (test_output[tok].float()
                                    - ref_output[tok].float()).abs().max()
                        print(f"  max_diff from ref: {max_diff}")
                        return False
                    else:
                        # Verify output matches reference
                        match = torch.allclose(test_output[tok],
                                               ref_output[tok])
                        if not match:
                            max_diff = (test_output[tok].float()
                                        - ref_output[tok].float()).abs().max()
                            print(f"WARN batch={batch_size} "
                                  f"pattern={nan_pattern} tok={tok}: "
                                  f"output differs from ref, "
                                  f"max_diff={max_diff}")

            print(f"OK batch={batch_size} pattern={nan_pattern}")

    print("\nAll tests passed — no cross-token NaN contamination")
    return True


if __name__ == "__main__":
    test_nan_contamination()
