# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Functional tests for Helix parallelism and standard DCP.

This test suite validates correctness of all four operational modes:
1. Standard DCP + GQA models
2. Standard DCP + MLA models  
3. Helix + GQA models
4. Helix + MLA models

Requirements:
- 4+ GPUs with compute capability 9.0+ (Hopper/Blackwell)
- ~40GB VRAM per GPU for the test models (8B models with TP=4)

=============================================================================
INSTALLATION & SETUP
=============================================================================

1. Clone the vLLM fork with Helix support:
   
   git clone https://github.com/sungsooha/vllm.git
   cd vllm
   git checkout helix-migration

2. Create a virtual environment (recommended):
   
   python -m venv venv
   source venv/bin/activate

3. Install vLLM in development/editable mode:
   
   pip install -e .
   
   Or for faster installation (skip building from source if wheels available):
   
   pip install -e ".[dev]"

4. Verify installation:
   
   python -c "import vllm; print(vllm.__version__)"

=============================================================================
RUNNING TESTS
=============================================================================

# Quick smoke test (fastest, ~2-3 min)
pytest tests/distributed/test_helix_functional.py::TestQuickSmoke -v -s

# All functional tests (~15-20 min)
pytest tests/distributed/test_helix_functional.py -v -s

# Run specific test classes:
pytest tests/distributed/test_helix_functional.py::TestHelixGQA -v -s
pytest tests/distributed/test_helix_functional.py::TestHelixMLA -v -s
pytest tests/distributed/test_helix_functional.py::TestStandardDCP -v -s
pytest tests/distributed/test_helix_functional.py::TestHelixVsDCPConsistency -v -s

# Run with specific GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/distributed/test_helix_functional.py -v -s

# Run with more verbose output:
pytest tests/distributed/test_helix_functional.py -v -s --tb=long

=============================================================================
"""

import os
from typing import NamedTuple

import pytest
import torch

from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
from vllm.logger import init_logger

logger = init_logger("test_helix_functional")

# =============================================================================
# Test Configuration
# =============================================================================

# Test models - using Llama-based models consistent with Nemotron testing
# - GQA model: Llama 3.1 8B (8 KV heads, 32 Q heads)
# - MLA model: DeepSeek-V2-Lite (MLA architecture with latent KV)
#
# Note: These are medium-sized models suitable for 4+ GPU testing on Hopper/Blackwell
# Adjust models based on available VRAM if needed.

# GQA Model options (pick one based on your setup):
# - "meta-llama/Llama-3.1-8B-Instruct"           # 8B, 8 KV heads, requires license
# - "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"       # 8B, 8 KV heads, NVIDIA variant
# - "meta-llama/Llama-3.2-3B-Instruct"           # 3B, 8 KV heads, smaller option
GQA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # num_kv_heads=8, num_q_heads=32

# MLA Model (DeepSeek with Multi-head Latent Attention)
MLA_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"  # MLA architecture

# Sanity check prompts - designed to detect gibberish
SANITY_PROMPTS = [
    # Simple math (deterministic answer)
    "What is 2 + 2? Answer with just the number:",
    # Logical reasoning
    "If all cats are animals, and Fluffy is a cat, then Fluffy is a(n):",
    # Common knowledge
    "The capital of France is:",
]

# Expected patterns in responses (should NOT be gibberish)
EXPECTED_PATTERNS = [
    ["4", "four"],
    ["animal"],
    ["Paris", "paris"],
]


class TestConfig(NamedTuple):
    """Configuration for a single test case."""
    tp_size: int
    dcp_size: int
    helix_mode: bool
    attn_backend: str
    model_type: str  # "gqa" or "mla"
    description: str


# =============================================================================
# Test Configurations Matrix
# =============================================================================

# GQA Model Configurations (Llama-3.1-8B: 32 Q heads, 8 KV heads)
# Helix constraints: TPA ≤ num_kv_heads (8), K % TPA == 0
GQA_CONFIGS = [
    # Standard DCP (helix_mode=False)
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=False,
        attn_backend="FLASH_ATTN",
        model_type="gqa",
        description="GQA + Standard DCP + FlashAttn",
    ),
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=False,
        attn_backend="FLASHINFER",
        model_type="gqa",
        description="GQA + Standard DCP + FlashInfer",
    ),
    # Helix mode (helix_mode=True)
    # TP=4, DCP=2 -> TPA=2, KVP=2
    # Valid for Llama-3.1-8B: TPA=2 ≤ 8 (num_kv_heads), 8 % 2 == 0
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=True,
        attn_backend="FLASH_ATTN",
        model_type="gqa",
        description="GQA + Helix (TPA=2, KVP=2) + FlashAttn",
    ),
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=True,
        attn_backend="FLASHINFER",
        model_type="gqa",
        description="GQA + Helix (TPA=2, KVP=2) + FlashInfer",
    ),
    # Additional config with larger TPA (if 8 GPUs available)
    # TP=8, DCP=2 -> TPA=4, KVP=2
    # TestConfig(
    #     tp_size=8,
    #     dcp_size=2,
    #     helix_mode=True,
    #     attn_backend="FLASH_ATTN",
    #     model_type="gqa",
    #     description="GQA + Helix (TPA=4, KVP=2) + FlashAttn",
    # ),
]

# MLA Model Configurations (DeepSeek-V2-Lite: MLA attention)
MLA_CONFIGS = [
    # Standard DCP (helix_mode=False)
    TestConfig(
        tp_size=4,
        dcp_size=4,
        helix_mode=False,
        attn_backend="FLASHMLA",
        model_type="mla",
        description="MLA + Standard DCP + FlashMLA",
    ),
    # Helix mode (helix_mode=True)
    # TP=4, DCP=4 -> TPA=1, KVP=4 (valid for MLA)
    TestConfig(
        tp_size=4,
        dcp_size=4,
        helix_mode=True,
        attn_backend="FLASHMLA",
        model_type="mla",
        description="MLA + Helix (TPA=1, KVP=4) + FlashMLA",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def check_gpu_requirements(tp_size: int, min_compute_capability: tuple = (9, 0)):
    """Check if GPU requirements are met, skip test if not."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs, have {num_gpus}")
    
    cc = torch.cuda.get_device_capability()
    if cc < min_compute_capability:
        pytest.skip(
            f"Need compute capability {min_compute_capability}, have {cc}"
        )


def validate_response(response_text: str, expected_patterns: list[str]) -> bool:
    """Check if response contains any of the expected patterns."""
    response_lower = response_text.lower()
    return any(pattern.lower() in response_lower for pattern in expected_patterns)


def is_gibberish(text: str) -> bool:
    """Heuristic check for gibberish output."""
    if not text or len(text.strip()) == 0:
        return True
    
    # Check for excessive special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if special_ratio > 0.5:
        return True
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:  # >80% repetition
            return True
    
    return False


def run_sanity_check(
    client,
    model_id: str,
    config: TestConfig,
) -> dict:
    """Run sanity check prompts and validate responses."""
    results = {
        "config": config.description,
        "passed": True,
        "responses": [],
        "errors": [],
    }
    
    for i, (prompt, expected) in enumerate(zip(SANITY_PROMPTS, EXPECTED_PATTERNS)):
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50,
                temperature=0,  # Deterministic
            )
            text = response.choices[0].text.strip()
            results["responses"].append(text)
            
            # Check for gibberish
            if is_gibberish(text):
                results["passed"] = False
                results["errors"].append(f"Prompt {i}: Gibberish detected: '{text}'")
                continue
            
            # Check for expected pattern
            if not validate_response(text, expected):
                # Not a failure, just a warning (model might phrase differently)
                logger.warning(
                    f"Prompt {i}: Expected one of {expected}, got '{text}'"
                )
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Prompt {i}: Exception: {e}")
    
    return results


def build_server_args(
    config: TestConfig,
    model_id: str,
    trust_remote_code: bool = False,
) -> list[str]:
    """Build vLLM server arguments for a test configuration."""
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
        "--tensor-parallel-size", str(config.tp_size),
        "--decode-context-parallel-size", str(config.dcp_size),
        "--attention-backend", config.attn_backend,
        "--enable-chunked-prefill",
    ]
    
    if config.helix_mode:
        args.append("--helix-mode")
    
    if trust_remote_code:
        args.append("--trust-remote-code")
    
    # KV cache interleave size for DCP
    # - MLA models benefit from larger interleave (64)
    # - GQA models use smaller interleave (16)
    if config.model_type == "mla":
        args.extend(["--cp-kv-cache-interleave-size", "64"])
    else:
        args.extend(["--cp-kv-cache-interleave-size", "16"])
    
    return args


def get_model_trust_remote_code(model_id: str) -> bool:
    """Determine if model requires trust_remote_code."""
    # Models that need trust_remote_code
    trust_required = [
        "deepseek",
        "DeepSeek",
    ]
    return any(name in model_id for name in trust_required)


# =============================================================================
# Test Classes
# =============================================================================

class TestHelixGQA:
    """Test Helix mode with GQA models (e.g., Llama, Nemotron)."""
    
    @pytest.mark.parametrize("config", [c for c in GQA_CONFIGS if c.helix_mode])
    @create_new_process_for_each_test()
    def test_helix_gqa_sanity(self, config: TestConfig, num_gpus_available):
        """Test Helix GQA mode produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        logger.info(f"Model: {GQA_MODEL}")
        
        trust_remote_code = get_model_trust_remote_code(GQA_MODEL)
        server_args = build_server_args(config, GQA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, GQA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestHelixMLA:
    """Test Helix mode with MLA models (e.g., DeepSeek-V2)."""
    
    @pytest.mark.parametrize("config", [c for c in MLA_CONFIGS if c.helix_mode])
    @create_new_process_for_each_test()
    def test_helix_mla_sanity(self, config: TestConfig, num_gpus_available):
        """Test Helix MLA mode produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        
        server_args = build_server_args(config, MLA_MODEL, trust_remote_code=True)
        
        with RemoteOpenAIServer(MLA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, MLA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestStandardDCP:
    """Test standard DCP mode (helix_mode=False) - baseline."""
    
    @pytest.mark.parametrize("config", [c for c in GQA_CONFIGS if not c.helix_mode])
    @create_new_process_for_each_test()
    def test_standard_dcp_gqa_sanity(self, config: TestConfig, num_gpus_available):
        """Test standard DCP with GQA model produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        logger.info(f"Model: {GQA_MODEL}")
        
        trust_remote_code = get_model_trust_remote_code(GQA_MODEL)
        server_args = build_server_args(config, GQA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, GQA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"
    
    @pytest.mark.parametrize("config", [c for c in MLA_CONFIGS if not c.helix_mode])
    @create_new_process_for_each_test()
    def test_standard_dcp_mla_sanity(self, config: TestConfig, num_gpus_available):
        """Test standard DCP with MLA model produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        logger.info(f"Model: {MLA_MODEL}")
        
        trust_remote_code = get_model_trust_remote_code(MLA_MODEL)
        server_args = build_server_args(config, MLA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(MLA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, MLA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestHelixVsDCPConsistency:
    """Test that Helix produces outputs consistent with standard DCP."""
    
    @create_new_process_for_each_test()
    def test_gqa_helix_vs_dcp_consistency(self, num_gpus_available):
        """Verify Helix GQA output matches standard DCP output."""
        check_gpu_requirements(4)
        
        logger.info(f"Model: {GQA_MODEL}")
        prompts = ["What is the capital of Japan?"]
        trust_remote_code = get_model_trust_remote_code(GQA_MODEL)
        
        # Standard DCP
        dcp_config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=False,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="baseline",
        )
        dcp_args = build_server_args(dcp_config, GQA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(GQA_MODEL, dcp_args, num_gpus=4) as server:
            client = server.get_client()
            dcp_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=GQA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                dcp_outputs.append(response.choices[0].text)
        
        # Helix
        helix_config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=True,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="helix",
        )
        helix_args = build_server_args(helix_config, GQA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(GQA_MODEL, helix_args, num_gpus=4) as server:
            client = server.get_client()
            helix_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=GQA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                helix_outputs.append(response.choices[0].text)
        
        # Compare
        for i, (dcp_out, helix_out) in enumerate(zip(dcp_outputs, helix_outputs)):
            logger.info(f"Prompt {i}:")
            logger.info(f"  DCP:   '{dcp_out}'")
            logger.info(f"  Helix: '{helix_out}'")
            
            # For exact match (both use exact attention, should be identical)
            assert dcp_out == helix_out, (
                f"Output mismatch for prompt {i}:\n"
                f"DCP: {dcp_out}\n"
                f"Helix: {helix_out}"
            )
    
    @create_new_process_for_each_test()
    def test_mla_helix_vs_dcp_consistency(self, num_gpus_available):
        """Verify Helix MLA output matches standard DCP output."""
        check_gpu_requirements(4)
        
        logger.info(f"Model: {MLA_MODEL}")
        prompts = ["What is 5 + 5?"]
        trust_remote_code = get_model_trust_remote_code(MLA_MODEL)
        
        # Standard DCP
        dcp_config = TestConfig(
            tp_size=4, dcp_size=4, helix_mode=False,
            attn_backend="FLASHMLA", model_type="mla",
            description="baseline",
        )
        dcp_args = build_server_args(dcp_config, MLA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(MLA_MODEL, dcp_args, num_gpus=4) as server:
            client = server.get_client()
            dcp_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=MLA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                dcp_outputs.append(response.choices[0].text)
        
        # Helix
        helix_config = TestConfig(
            tp_size=4, dcp_size=4, helix_mode=True,
            attn_backend="FLASHMLA", model_type="mla",
            description="helix",
        )
        helix_args = build_server_args(helix_config, MLA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(MLA_MODEL, helix_args, num_gpus=4) as server:
            client = server.get_client()
            helix_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=MLA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                helix_outputs.append(response.choices[0].text)
        
        # Compare
        for i, (dcp_out, helix_out) in enumerate(zip(dcp_outputs, helix_outputs)):
            logger.info(f"Prompt {i}:")
            logger.info(f"  DCP:   '{dcp_out}'")
            logger.info(f"  Helix: '{helix_out}'")
            
            assert dcp_out == helix_out, (
                f"Output mismatch for prompt {i}:\n"
                f"DCP: {dcp_out}\n"
                f"Helix: {helix_out}"
            )


# =============================================================================
# Quick Smoke Test
# =============================================================================

class TestQuickSmoke:
    """Quick smoke test to verify basic functionality."""
    
    @create_new_process_for_each_test()
    def test_helix_gqa_quick(self, num_gpus_available):
        """Quick smoke test for Helix GQA."""
        check_gpu_requirements(4)
        
        logger.info(f"Model: {GQA_MODEL}")
        
        config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=True,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="Helix GQA smoke test",
        )
        
        trust_remote_code = get_model_trust_remote_code(GQA_MODEL)
        server_args = build_server_args(config, GQA_MODEL, trust_remote_code)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=4) as server:
            client = server.get_client()
            response = client.completions.create(
                model=GQA_MODEL,
                prompt="Hello, my name is",
                max_tokens=20,
                temperature=0,
            )
            text = response.choices[0].text
            logger.info(f"Response: '{text}'")
            assert not is_gibberish(text), f"Gibberish output: {text}"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
