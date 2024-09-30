from typing import Dict, List, Optional

from ..utils import compare_all_settings


def test_compile_correctness():
    all_args = [
        ["--enforce-eager"],
        ["--enforce-eager"],
        ["--enforce-eager"],
    ]
    all_envs: List[Optional[Dict[str, str]]] = [{
        "VLLM_TORCH_COMPILE_LEVEL":
        str(i)
    } for i in range(3)]
    compare_all_settings("meta-llama/Meta-Llama-3-8B", all_args, all_envs)
