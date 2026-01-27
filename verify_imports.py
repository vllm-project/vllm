
import sys
import os

# Add the current directory to sys.path to ensure we can import vllm
sys.path.append(os.getcwd())

print("Testing imports...")

try:
    print("1. Importing from new unified module 'vllm.utils.flashinfer'...")
    from vllm.utils.flashinfer import (
        FlashinferMoeBackend,
        get_flashinfer_moe_backend,
        prepare_fp8_moe_layer_for_fi
    )
    print("   -> Success! (Lazy import worked)")
except Exception as e:
    print(f"   -> FAILED: {e}")
    sys.exit(1)

try:
    print("2. Importing from deprecated module to check backward compatibility...")
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        FlashinferMoeBackend as BackendOld,
    )
    print("   -> Success! (Re-export worked)")
except Exception as e:
    print(f"   -> FAILED: {e}")
    sys.exit(1)

print("\nALL IMPORT TESTS PASSED.")
