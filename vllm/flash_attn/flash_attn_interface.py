import json
from typing import Dict, Any


def get_fused_attn_config(model: str, dtype: str, use_fp4: bool) -> Dict[str, Any]:
    """Return the compilation configuration for flash attention.

    For fp4 quantization we observed that the fused kernel is slower than the
    unfused version, so we explicitly disable the fusion pass in that case.
    """
    # Default configuration – fusion is enabled for non‑fp4 paths
    config = {
        "compilation_config": {
            "use_inductor_graph_partition": True,
            "pass_config": {
                "fuse_attn_quant": True  # enabled by default
            }
        }
    }

    # When fp4 is requested, turn off the fused attention quantization
    # because the current fused kernel underperforms the unfused implementation.
    if use_fp4:
        config["compilation_config"]["pass_config"]["fuse_attn_quant"] = False

    return config


# ---------------------------------------------------------------------------
# Helper functions used by the benchmarking script
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--use_fp4", action="store_true", help="Enable fp4 quantization")
    args = parser.parse_args()

    cfg = get_fused_attn_config(args.model, "fp8", args.use_fp4)
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
