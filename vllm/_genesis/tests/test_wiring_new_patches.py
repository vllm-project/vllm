# SPDX-License-Identifier: Apache-2.0
"""TDD for the new Phase-3-wave-2 wiring modules:
  - patch_3_tq_bf16_cast
  - patch_6_tq_block_size_align
  - patch_15_qwen3_none_null

Each tests: apply to synthetic baseline file, idempotency, non-NVIDIA skip
(where relevant), upstream-drift skip.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────────────────────
#                             P3 — TQ BF16->FP8
# ────────────────────────────────────────────────────────────────────────

_P3_BASELINE = """# SPDX-License-Identifier: Apache-2.0
# some triton store kernel

import triton.language as tl

@triton.jit
def _store_quantized_key(Key_ptr, KV_cache_ptr, base, d_offs, d_mask, FP8_E4B15):
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)
    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
"""


@pytest.fixture
def fake_tq_store(tmp_path, monkeypatch):
    path = tmp_path / "triton_turboquant_store.py"
    path.write_text(_P3_BASELINE)

    from vllm._genesis.wiring.legacy import patch_3_tq_bf16_cast as p3
    monkeypatch.setattr(p3, "resolve_vllm_file",
                        lambda rel: str(path) if "triton_turboquant_store" in rel else None)
    monkeypatch.setattr(p3, "vllm_install_root", lambda: "/fake")
    monkeypatch.setattr(p3, "is_nvidia_cuda", lambda: True)
    monkeypatch.setattr(p3, "is_sm_at_least", lambda *a, **kw: True)
    return str(path)


class TestPatch3:
    def test_apply_writes_fp16_intermediate(self, fake_tq_store):
        from vllm._genesis.wiring.legacy import patch_3_tq_bf16_cast as p3
        status, reason = p3.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_tq_store).read()
        assert "tl.float16).to(tl.float8e4b15)" in content
        assert "Genesis P3 TQ BF16->FP8 Ampere fix" in content

    def test_idempotent(self, fake_tq_store):
        from vllm._genesis.wiring.legacy import patch_3_tq_bf16_cast as p3
        s1, _ = p3.apply()
        s2, _ = p3.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_tq_store).read()
        # marker appears once
        assert content.count("Genesis P3 TQ BF16->FP8 Ampere fix v7.0") == 1

    def test_skip_on_non_nvidia(self, fake_tq_store, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_3_tq_bf16_cast as p3
        monkeypatch.setattr(p3, "is_nvidia_cuda", lambda: False)
        status, reason = p3.apply()
        assert status == "skipped"
        assert "non-NVIDIA" in reason


# ────────────────────────────────────────────────────────────────────────
#                             P6 — TQ block size
# ────────────────────────────────────────────────────────────────────────

_P6_BASELINE = """# SPDX-License-Identifier: Apache-2.0

class Platform:
    def get_attn_block_size(self, vllm_config):
        from vllm.v1.attention.backend import MultipleOf
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MLAAttentionSpec,
            get_kv_quant_mode,
        )

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        kv_quant_mode = get_kv_quant_mode(cache_config.cache_dtype)

        if model_config.use_mla:
            attn_page_size_1_token = MLAAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
            ).page_size_bytes
        else:
            attn_page_size_1_token = FullAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
            ).page_size_bytes

        return attn_page_size_1_token
"""


@pytest.fixture
def fake_interface(tmp_path, monkeypatch):
    path = tmp_path / "interface.py"
    path.write_text(_P6_BASELINE)

    from vllm._genesis.wiring.legacy import patch_6_tq_block_size_align as p6
    monkeypatch.setattr(p6, "resolve_vllm_file",
                        lambda rel: str(path) if "interface.py" in rel else None)
    monkeypatch.setattr(p6, "vllm_install_root", lambda: "/fake")
    monkeypatch.setattr(p6, "is_nvidia_cuda", lambda: True)
    return str(path)


class TestPatch6:
    def test_apply_adds_tq_branch(self, fake_interface):
        from vllm._genesis.wiring.legacy import patch_6_tq_block_size_align as p6
        status, reason = p6.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_interface).read()
        assert "TQFullAttentionSpec" in content
        assert 'turboquant_' in content
        assert "Genesis P6 TQ-aware block size alignment" in content

    def test_idempotent(self, fake_interface):
        from vllm._genesis.wiring.legacy import patch_6_tq_block_size_align as p6
        s1, _ = p6.apply()
        s2, _ = p6.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_interface).read()
        # TQFullAttentionSpec import added once
        assert content.count("TQFullAttentionSpec,  # [Genesis P6]") == 1

    def test_skip_on_non_nvidia(self, fake_interface, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_6_tq_block_size_align as p6
        monkeypatch.setattr(p6, "is_nvidia_cuda", lambda: False)
        status, reason = p6.apply()
        assert status == "skipped"
        assert "non-NVIDIA" in reason

    def test_skip_when_upstream_fully_merged(self, fake_interface):
        """If TQFullAttentionSpec is already imported → upstream merged #39931."""
        from vllm._genesis.wiring.legacy import patch_6_tq_block_size_align as p6
        path = fake_interface
        content = open(path).read()
        # Inject upstream marker
        open(path, "w").write(content.replace(
            "MLAAttentionSpec,\n            get_kv_quant_mode,",
            "MLAAttentionSpec,\n            TQFullAttentionSpec,\n            get_kv_quant_mode,",
        ))
        status, reason = p6.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()


# ────────────────────────────────────────────────────────────────────────
#                         P15 — Qwen3 None/null
# ────────────────────────────────────────────────────────────────────────

_P15_BASELINE = """# SPDX-License-Identifier: Apache-2.0
# qwen3coder tool parser

def _convert_param_value(param_value, param_type):
    if param_value is None:
        return None
    if isinstance(param_value, str):
        # Handle null value for any type
        if param_value.lower() == "null":
            return None
    return param_value
"""


@pytest.fixture
def fake_qwen3_parser(tmp_path, monkeypatch):
    path = tmp_path / "qwen3coder_tool_parser.py"
    path.write_text(_P15_BASELINE)

    from vllm._genesis.wiring.legacy import patch_15_qwen3_none_null as p15
    monkeypatch.setattr(p15, "resolve_vllm_file",
                        lambda rel: str(path) if "qwen3coder_tool_parser" in rel else None)
    monkeypatch.setattr(p15, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch15:
    def test_apply_accepts_none(self, fake_qwen3_parser):
        from vllm._genesis.wiring.legacy import patch_15_qwen3_none_null as p15
        status, reason = p15.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_qwen3_parser).read()
        assert '("null", "none")' in content
        assert "[Genesis P15]" in content

    def test_idempotent(self, fake_qwen3_parser):
        from vllm._genesis.wiring.legacy import patch_15_qwen3_none_null as p15
        s1, _ = p15.apply()
        s2, _ = p15.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_qwen3_parser).read()
        assert content.count("Genesis P15 Qwen3 None/null") == 1

    def test_skip_when_upstream_merged(self, fake_qwen3_parser):
        from vllm._genesis.wiring.legacy import patch_15_qwen3_none_null as p15
        path = fake_qwen3_parser
        # Simulate upstream form by directly patching in the tuple check
        content = open(path).read().replace(
            '"null"',
            '("null", "none")',
        )
        open(path, "w").write(content)
        status, reason = p15.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()


# ────────────────────────────────────────────────────────────────────────
#                      P23 — Marlin FP32_REDUCE env
# ────────────────────────────────────────────────────────────────────────

class TestPatch23Env:
    def test_explicit_env_true(self, monkeypatch):
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        monkeypatch.setenv("VLLM_MARLIN_FP32_REDUCE", "1")
        assert m.get_fp32_reduce_override() is True
        assert m.should_disable_fp32_reduce() is False

    def test_explicit_env_false(self, monkeypatch):
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        monkeypatch.setenv("VLLM_MARLIN_FP32_REDUCE", "0")
        assert m.get_fp32_reduce_override() is False
        assert m.should_disable_fp32_reduce() is True

    def test_auto_ampere_disables(self, monkeypatch):
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_FP32_REDUCE", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)

        def sm_at_least(major, minor=0):
            return (major, minor) <= (8, 6)  # Ampere range
        monkeypatch.setattr(guards, "is_sm_at_least", sm_at_least)
        assert m.should_disable_fp32_reduce() is True  # Ampere → disabled

    def test_auto_hopper_keeps(self, monkeypatch):
        """Hopper SM>=9.0: native FP32 tensor cores → keep default (don't disable)."""
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_FP32_REDUCE", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least", lambda *a, **kw: True)
        assert m.should_disable_fp32_reduce() is False

    def test_auto_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_FP32_REDUCE", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert m.should_disable_fp32_reduce() is False

    def test_invalid_env_falls_to_auto(self, monkeypatch):
        from vllm._genesis.kernels import marlin_fp32_reduce as m
        monkeypatch.setenv("VLLM_MARLIN_FP32_REDUCE", "maybe")
        assert m.get_fp32_reduce_override() is None


# ────────────────────────────────────────────────────────────────────────
#          P12 — Qwen3 <tool_call> implicit reasoning end
# ────────────────────────────────────────────────────────────────────────

_P12_BASELINE = '''# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class Qwen3ReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    @property
    def start_token(self):
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(self, model_output, request):
        return None, model_output
'''


@pytest.fixture
def fake_qwen3_reasoning_p12(tmp_path, monkeypatch):
    d = tmp_path / "reasoning"
    d.mkdir()
    path = d / "qwen3_reasoning_parser.py"
    path.write_text(_P12_BASELINE)

    from vllm._genesis.wiring.legacy import patch_12_tool_call_reasoning as p12
    monkeypatch.setattr(
        p12, "resolve_vllm_file",
        lambda rel: str(path) if "qwen3_reasoning_parser" in rel else None,
    )
    monkeypatch.setattr(p12, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch12:
    def test_apply_adds_tokens_and_hooks(self, fake_qwen3_reasoning_p12):
        from vllm._genesis.wiring.legacy import patch_12_tool_call_reasoning as p12
        status, reason = p12.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_qwen3_reasoning_p12).read()
        assert "_tool_call_token_id" in content
        assert "_tool_call_end_token_id" in content
        assert "def is_reasoning_end(self" in content
        assert "def is_reasoning_end_streaming" in content
        assert "def extract_content_ids" in content
        assert "[Genesis P12]" in content

    def test_idempotent(self, fake_qwen3_reasoning_p12):
        from vllm._genesis.wiring.legacy import patch_12_tool_call_reasoning as p12
        s1, _ = p12.apply()
        s2, _ = p12.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_qwen3_reasoning_p12).read()
        assert content.count(
            "Genesis P12 Qwen3 <tool_call> implicit reasoning end v7.0"
        ) == 1

    def test_patched_file_is_valid_python(self, fake_qwen3_reasoning_p12):
        import ast
        from vllm._genesis.wiring.legacy import patch_12_tool_call_reasoning as p12
        p12.apply()
        ast.parse(open(fake_qwen3_reasoning_p12).read())

    def test_upstream_drift_skip(self, fake_qwen3_reasoning_p12):
        from vllm._genesis.wiring.legacy import patch_12_tool_call_reasoning as p12
        open(fake_qwen3_reasoning_p12, "a").write(
            "\n# _tool_call_token_id upstream merged\n"
        )
        status, reason = p12.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()

    def test_coexists_with_p27(self, fake_qwen3_reasoning_p12):
        """P12 applied first; P27's non-conflicting anchors still apply."""
        from vllm._genesis.wiring.legacy import (
            patch_12_tool_call_reasoning as p12,
            patch_27_reasoning_before_think as p27,
        )

        # Rewrite fake to include both P12 and P27 anchors. Use baseline
        # with both minimal stubs.
        baseline = '''# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class Qwen3ReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    @property
    def start_token(self):
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(self, model_output, request):
        """Extract reasoning."""
        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )
        if self.end_token not in model_output:
            if not self.thinking_enabled:
                return None, model_output
            return model_output, None

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids,
    ):
        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]
        return None
'''
        open(fake_qwen3_reasoning_p12, "w").write(baseline)

        # Apply both in order
        s12, r12 = p12.apply()
        assert s12 == "applied", f"P12: {r12}"

        # Patch P27 fixture pointer to the same file
        # Re-monkeypatch p27 resolve
        orig_resolve = p27.resolve_vllm_file
        p27.resolve_vllm_file = (
            lambda rel: fake_qwen3_reasoning_p12
            if "qwen3_reasoning_parser" in rel else None
        )
        p27.vllm_install_root = lambda: "/fake"
        try:
            s27, r27 = p27.apply()
            assert s27 == "applied", f"P27 after P12: {r27}"
        finally:
            p27.resolve_vllm_file = orig_resolve

        import ast
        ast.parse(open(fake_qwen3_reasoning_p12).read())
        combined = open(fake_qwen3_reasoning_p12).read()
        assert "Genesis P12 Qwen3 <tool_call>" in combined
        assert "Genesis P27 Qwen3 BEFORE-THINK fallback v7.0" in combined
        assert "_tool_call_token_id" in combined
        assert "_genesis_before_think" in combined


# ────────────────────────────────────────────────────────────────────────
#                    P27 — Qwen3 BEFORE-THINK fallback
# ────────────────────────────────────────────────────────────────────────

_P27_BASELINE = '''# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class Qwen3ReasoningParser:
    def extract_reasoning(self, model_output, request):
        """Extract reasoning."""

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            if not self.thinking_enabled:
                return None, model_output
            return model_output, None

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids,
    ):
        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            return None
        return None
'''


@pytest.fixture
def fake_qwen3_reasoning(tmp_path, monkeypatch):
    d = tmp_path / "reasoning"
    d.mkdir()
    path = d / "qwen3_reasoning_parser.py"
    path.write_text(_P27_BASELINE)

    from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
    monkeypatch.setattr(
        p27, "resolve_vllm_file",
        lambda rel: str(path) if "qwen3_reasoning_parser" in rel else None,
    )
    monkeypatch.setattr(p27, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch27:
    def test_apply_writes_before_think_capture(self, fake_qwen3_reasoning):
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        status, reason = p27.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_qwen3_reasoning).read()
        # Non-streaming capture
        assert "_genesis_before_think" in content
        # Non-streaming prepend to content
        assert "if _genesis_before_think:" in content
        # Streaming emit
        assert "_genesis_pre_think_content" in content
        assert "[Genesis P27]" in content

    def test_idempotent(self, fake_qwen3_reasoning):
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        s1, _ = p27.apply()
        s2, _ = p27.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_qwen3_reasoning).read()
        assert content.count("Genesis P27 Qwen3 BEFORE-THINK fallback v7.0") == 1

    def test_skip_when_upstream_merged(self, fake_qwen3_reasoning):
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        # Inject an upstream drift marker
        original = open(fake_qwen3_reasoning).read()
        open(fake_qwen3_reasoning, "w").write(
            original.replace(
                "# Strip <think>",
                "# Strip <think>  (before_think captured upstream)",
                1,
            )
        )
        status, reason = p27.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()

    def test_apply_produces_valid_python(self, fake_qwen3_reasoning):
        """The resulting file must still be valid Python."""
        import ast
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        status, _ = p27.apply()
        assert status == "applied"
        content = open(fake_qwen3_reasoning).read()
        # Strip the marker comment line (it's a `#` comment so ast handles it)
        ast.parse(content)

    def test_skip_when_file_missing(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        monkeypatch.setattr(p27, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(p27, "vllm_install_root", lambda: "/fake")
        status, _reason = p27.apply()
        assert status == "skipped"

# ────────────────────────────────────────────────────────────────────────
#                    P7 — GDN dual-stream in_proj
# ────────────────────────────────────────────────────────────────────────

_P7_BASELINE = '''# SPDX-License-Identifier: Apache-2.0
import torch


class GatedDeltaNet:
    def forward_cuda(self, hidden_states, output):
        """forward_cuda: two paths, LoRA and non-LoRA."""
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            # downstream processing
        return mixed_qkvz, ba
'''


@pytest.fixture
def fake_gdn_linear_attn(tmp_path, monkeypatch):
    d = tmp_path / "model_executor" / "layers" / "mamba"
    d.mkdir(parents=True)
    path = d / "gdn_linear_attn.py"
    path.write_text(_P7_BASELINE)

    from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
    monkeypatch.setattr(
        p7, "resolve_vllm_file",
        lambda rel: str(path) if "gdn_linear_attn" in rel else None,
    )
    monkeypatch.setattr(p7, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch7Deferred:
    """Post-2026-04-24: P7 is deferred by default because CUDA streams
    are incompatible with torch.compile's aot_compile_fullgraph. Re-enable
    via env GENESIS_ENABLE_P7=1 (eager-only).

    We test the TWO behaviours:
      (a) default (no env): apply() returns 'skipped' with explicit reason
      (b) env set: apply() performs the text-patch as before
    """

    def test_default_skips_with_explicit_reason(
        self, fake_gdn_linear_attn, monkeypatch,
    ):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.delenv("GENESIS_ENABLE_P7", raising=False)
        status, reason = p7.apply()
        assert status == "skipped"
        assert "deferred" in reason.lower()
        assert "aot_compile" in reason.lower() or "fullgraph" in reason.lower()

    def test_env_enabled_applies(self, fake_gdn_linear_attn, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        status, reason = p7.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_gdn_linear_attn).read()
        assert "DualStreamDispatcher.maybe_parallel" in content
        assert "[Genesis P7]" in content

    def test_env_enabled_only_non_lora_branch(
        self, fake_gdn_linear_attn, monkeypatch,
    ):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        p7.apply()
        content = open(fake_gdn_linear_attn).read()
        assert "mixed_qkv, _ = self.in_proj_qkv(hidden_states)" in content
        assert content.count("mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)") == 0

    def test_env_enabled_idempotent(self, fake_gdn_linear_attn, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        s1, _ = p7.apply()
        s2, _ = p7.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_gdn_linear_attn).read()
        assert content.count("Genesis P7 GDN dual-stream in_proj v7.0") == 1

    def test_env_enabled_patched_file_valid_python(
        self, fake_gdn_linear_attn, monkeypatch,
    ):
        import ast
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        p7.apply()
        ast.parse(open(fake_gdn_linear_attn).read())

    def test_env_enabled_upstream_drift_detected(
        self, fake_gdn_linear_attn, monkeypatch,
    ):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        content = open(fake_gdn_linear_attn).read()
        open(fake_gdn_linear_attn, "w").write(
            "# DualStreamDispatcher upstream merged\n" + content
        )
        status, reason = p7.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()

    def test_env_enabled_skip_when_file_missing(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        monkeypatch.setenv("GENESIS_ENABLE_P7", "1")
        monkeypatch.setattr(p7, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(p7, "vllm_install_root", lambda: "/fake")
        status, _reason = p7.apply()
        assert status == "skipped"


# ────────────────────────────────────────────────────────────────────────
#                    P28 — GDN core_attn_out prealloc
# ────────────────────────────────────────────────────────────────────────

_P28_BASELINE = '''# SPDX-License-Identifier: Apache-2.0
import torch


class GatedDeltaNet:
    num_v_heads = 32
    tp_size = 1
    head_v_dim = 128

    def forward_cuda(self, hidden_states, output):
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return core_attn_out

    def forward_xpu(self, hidden_states, output):
        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return core_attn_out
'''


@pytest.fixture
def fake_gdn_linear_attn_p28(tmp_path, monkeypatch):
    d = tmp_path / "model_executor" / "layers" / "mamba"
    d.mkdir(parents=True)
    path = d / "gdn_linear_attn.py"
    path.write_text(_P28_BASELINE)
    from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
    monkeypatch.setattr(
        p28, "resolve_vllm_file",
        lambda rel: str(path) if "gdn_linear_attn" in rel else None,
    )
    monkeypatch.setattr(p28, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch28:
    def test_apply_rewires_forward_cuda_only(self, fake_gdn_linear_attn_p28):
        from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
        status, reason = p28.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_gdn_linear_attn_p28).read()
        # forward_cuda's torch.zeros replaced with pure tensor slice logic
        assert "_genesis_gdn_core_attn_buf" in content
        assert "[:num_tokens].zero_()" in content
        assert "[Genesis P28]" in content
        # forward_xpu's torch.zeros PRESERVED (our anchor disambiguates
        # via the #28182 comment which is unique to forward_cuda)
        assert "    def forward_xpu" in content

    def test_idempotent(self, fake_gdn_linear_attn_p28):
        from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
        s1, _ = p28.apply()
        s2, _ = p28.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_gdn_linear_attn_p28).read()
        assert content.count(
            "Genesis P28 GDN core_attn_out prealloc v7.0"
        ) == 1

    def test_patched_file_is_valid_python(self, fake_gdn_linear_attn_p28):
        import ast
        from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
        p28.apply()
        ast.parse(open(fake_gdn_linear_attn_p28).read())

    def test_upstream_drift_detected(self, fake_gdn_linear_attn_p28):
        from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
        content = open(fake_gdn_linear_attn_p28).read()
        # Use the CURRENT drift marker (matches patch_28_gdn_core_attn.UPSTREAM_DRIFT_MARKERS)
        open(fake_gdn_linear_attn_p28, "w").write(
            "# gdn_core_attn_out_buffer upstream merged\n" + content
        )
        status, reason = p28.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()

    def test_skip_when_file_missing(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_28_gdn_core_attn as p28
        monkeypatch.setattr(p28, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(p28, "vllm_install_root", lambda: "/fake")
        status, _reason = p28.apply()
        assert status == "skipped"


# ────────────────────────────────────────────────────────────────────────
#             P34 — Mamba zero-collapse deadlock guard
# ────────────────────────────────────────────────────────────────────────

_P34_BASELINE = '''# SPDX-License-Identifier: Apache-2.0
"""Fake scheduler with the EXACT indentation of real scheduler.py:
- class (0) → method (4) → body (8) → outer-if (8) → inner-if (12) → assignment (16).
The P34 anchor targets the innermost 3 lines, at 12/16/16-space prefix."""


class Scheduler:
    def _mamba_block_aligned_split(self, request, num_new_tokens, num_computed_tokens):
        # Outer scope mimics the real scheduler.py's outer `if` guard.
        if True:
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens
'''


@pytest.fixture
def fake_scheduler_p34(tmp_path, monkeypatch):
    d = tmp_path / "v1" / "core" / "sched"
    d.mkdir(parents=True)
    path = d / "scheduler.py"
    path.write_text(_P34_BASELINE)

    from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
    monkeypatch.setattr(
        p34, "resolve_vllm_file",
        lambda rel: str(path) if "scheduler.py" in rel else None,
    )
    monkeypatch.setattr(p34, "vllm_install_root", lambda: "/fake")
    return str(path)


class TestPatch34MambaDeadlock:
    def test_apply_inserts_aligned_guard(self, fake_scheduler_p34):
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        status, reason = p34.apply()
        assert status == "applied", f"{status}: {reason}"
        content = open(fake_scheduler_p34).read()
        # Original single-line alignment is replaced by aligned-intermediate + guard
        assert "aligned = num_new_tokens // block_size * block_size" in content
        assert "if aligned > 0:" in content
        assert "[Genesis P34]" in content

    def test_idempotent(self, fake_scheduler_p34):
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        s1, _ = p34.apply()
        s2, _ = p34.apply()
        assert s1 == "applied" and s2 == "applied"
        content = open(fake_scheduler_p34).read()
        assert content.count(
            "Genesis P34 Mamba zero-collapse deadlock guard v7.0"
        ) == 1

    def test_patched_file_is_valid_python(self, fake_scheduler_p34):
        import ast
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        p34.apply()
        ast.parse(open(fake_scheduler_p34).read())

    def test_upstream_drift_pr40757_detected(self, fake_scheduler_p34):
        """Simulate PR #40757 landing: the aligned= pattern appears before
        our patch runs. We must self-retire."""
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        # Prepend a line that looks like the PR #40757 fix had already merged
        original = open(fake_scheduler_p34).read()
        open(fake_scheduler_p34, "w").write(
            "# upstream fix landed\n"
            "aligned = num_new_tokens // block_size * block_size\n"
            + original
        )
        status, reason = p34.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower()

    def test_semantic_fix_correct(self, fake_scheduler_p34):
        """Apply P34 then exec the patched scheduler and assert that the
        alignment does NOT collapse num_new_tokens to 0."""
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        p34.apply()
        patched = open(fake_scheduler_p34).read()

        ns = {}
        exec(compile(patched, fake_scheduler_p34, "exec"), ns)
        sched_cls = ns["Scheduler"]
        sched = sched_cls()

        # Craft a scenario that would have collapsed: block_size=16,
        # num_new_tokens=10 (< block_size), and the aligned branch fires.
        class Req:
            num_tokens = 100
        class Cfg:
            block_size = 16

        sched.cache_config = Cfg()
        sched.use_eagle = False

        result = sched._mamba_block_aligned_split(
            Req(),
            num_new_tokens=10,   # would collapse to 0 after alignment
            num_computed_tokens=0,
        )
        # Pre-fix: 10 // 16 * 16 = 0 → deadlock.
        # Post-fix: aligned=0, keep original → 10.
        assert result == 10, f"Deadlock not fixed: got {result}"

    def test_semantic_non_deadlock_path_unchanged(self, fake_scheduler_p34):
        """When alignment is non-zero the behaviour matches upstream."""
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        p34.apply()
        patched = open(fake_scheduler_p34).read()
        ns = {}
        exec(compile(patched, fake_scheduler_p34, "exec"), ns)
        sched = ns["Scheduler"]()

        class Req:
            num_tokens = 100
        class Cfg:
            block_size = 16
        sched.cache_config = Cfg()
        sched.use_eagle = False

        # 47 // 16 * 16 = 32 (aligned > 0) → behaviour unchanged
        result = sched._mamba_block_aligned_split(
            Req(), num_new_tokens=47, num_computed_tokens=0,
        )
        assert result == 32

    def test_skip_when_file_missing(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_34_mamba_deadlock_guard as p34
        monkeypatch.setattr(p34, "resolve_vllm_file", lambda rel: None)
        monkeypatch.setattr(p34, "vllm_install_root", lambda: "/fake")
        status, _reason = p34.apply()
        assert status == "skipped"


    def test_non_streaming_before_think_preserved_after_apply(
        self, fake_qwen3_reasoning,
    ):
        """Behavioral test: after patching, the patched parser should
        preserve BEFORE-THINK text in content on non-streaming extraction."""
        from vllm._genesis.wiring.legacy import patch_27_reasoning_before_think as p27
        p27.apply()

        # Simulate the patched module by execing the content in a namespace
        # with stubbed imports.
        patched = open(fake_qwen3_reasoning).read()

        ns: dict = {}
        # Stub the DeltaMessage import
        import sys, types
        fake_proto = types.ModuleType(
            "vllm.entrypoints.openai.engine.protocol"
        )
        fake_proto.DeltaMessage = lambda **kw: ("DeltaMessage", kw)
        fake_entry = types.ModuleType("vllm.entrypoints")
        fake_oai = types.ModuleType("vllm.entrypoints.openai")
        fake_engine = types.ModuleType("vllm.entrypoints.openai.engine")
        sys.modules["vllm.entrypoints"] = fake_entry
        sys.modules["vllm.entrypoints.openai"] = fake_oai
        sys.modules["vllm.entrypoints.openai.engine"] = fake_engine
        sys.modules["vllm.entrypoints.openai.engine.protocol"] = fake_proto

        exec(compile(patched, fake_qwen3_reasoning, "exec"), ns)
        parser = ns["Qwen3ReasoningParser"]()
        parser.start_token = "<think>"
        parser.end_token = "</think>"
        parser.thinking_enabled = True

        reasoning, content = parser.extract_reasoning(
            "prefix-content <think>thinking here</think>answer", None,
        )
        assert reasoning == "thinking here"
        # Before-think prefix must be preserved in content
        assert content is not None
        assert "prefix-content" in content
        assert "answer" in content
