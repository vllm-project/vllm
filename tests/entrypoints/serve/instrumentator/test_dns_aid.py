"""Unit tests for vllm/entrypoints/serve/instrumentator/dns_aid.py.

All dns_aid library calls are mocked -- no real DNS is touched.
"""

import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import vllm.entrypoints.serve.instrumentator.dns_aid as dns_aid_mod
from vllm.entrypoints.serve.instrumentator.dns_aid import (
    _MAX_DNS_LABEL_LEN,
    build_agent_record,
    resolve_target_hostname,
    slugify,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_dns_aid() -> ModuleType:
    """Return a minimal fake dns_aid module."""
    mod = ModuleType("dns_aid")

    class SvcParams:
        def __init__(self, *, alpn, hints):
            self.alpn = alpn
            self.hints = hints

    class AgentRecord:
        def __init__(self, *, name, target, port, params, ttl):
            self.name = name
            self.target = target
            self.port = port
            self.params = params
            self.ttl = ttl

    mod.SvcParams = SvcParams
    mod.AgentRecord = AgentRecord
    mod.register_agent = MagicMock()
    mod.deregister_agent = MagicMock()
    return mod


def _make_engine_client(
    *,
    model: str = "meta-llama/Llama-3-70b-instruct",
    max_model_len: int = 131072,
    quantization: str = "fp8",
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
) -> MagicMock:
    client = MagicMock()
    client.model_config.model = model
    client.model_config.max_model_len = max_model_len
    client.model_config.quantization = quantization
    client.vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    client.vllm_config.parallel_config.tensor_parallel_size = tensor_parallel_size
    return client


def _make_args(**kwargs) -> Namespace:
    defaults = dict(
        dns_aid_enabled=True,
        dns_aid_zone="example.internal",
        dns_aid_name=None,
        host="serve.example.internal",
        port=8000,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dns_aid():
    """Inject a fake dns_aid module into sys.modules for the test duration."""
    fake = _make_fake_dns_aid()
    orig = sys.modules.get("dns_aid")
    sys.modules["dns_aid"] = fake
    yield fake
    if orig is None:
        sys.modules.pop("dns_aid", None)
    else:
        sys.modules["dns_aid"] = orig


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_slash(self):
        expected = "meta-llama-llama-3-70b-instruct"
        assert slugify("meta-llama/Llama-3-70b-instruct") == expected

    def test_dots(self):
        assert slugify("Qwen/Qwen3-0.6B") == "qwen-qwen3-0-6b"

    def test_consecutive_separators(self):
        assert slugify("org//model__v1") == "org-model-v1"

    def test_strips_leading_trailing(self):
        assert slugify("/model/") == "model"

    def test_long_name_truncated_with_hash(self):
        long_name = "org/" + "a" * 100 + "-instruct"
        slug = slugify(long_name)
        assert len(slug) <= _MAX_DNS_LABEL_LEN
        # Hash suffix should be 8 hex chars
        assert slug[-8:].isalnum()

    def test_exact_63_chars_not_truncated(self):
        name = "a" * _MAX_DNS_LABEL_LEN
        assert slugify(name) == name

    def test_truncation_preserves_uniqueness(self):
        name_a = "org/" + "a" * 100 + "-variant-alpha"
        name_b = "org/" + "a" * 100 + "-variant-beta"
        assert slugify(name_a) != slugify(name_b)


# ---------------------------------------------------------------------------
# resolve_target_hostname
# ---------------------------------------------------------------------------


class TestResolveTargetHostname:
    _FQDN_PATCH = "vllm.entrypoints.serve.instrumentator.dns_aid.socket.getfqdn"

    def test_fqdn_returned(self):
        host = "serve.example.internal"
        assert resolve_target_hostname(host) == host

    def test_none_falls_back(self):
        with patch(self._FQDN_PATCH, return_value="box.local"):
            assert resolve_target_hostname(None) == "box.local"

    def test_wildcard_ipv4_falls_back(self):
        with patch(self._FQDN_PATCH, return_value="box.local"):
            assert resolve_target_hostname("0.0.0.0") == "box.local"

    def test_ipv6_falls_back(self):
        with patch(self._FQDN_PATCH, return_value="box.local"):
            assert resolve_target_hostname("::") == "box.local"

    def test_ipv4_with_dots_falls_back(self):
        with patch(self._FQDN_PATCH, return_value="box.local"):
            assert resolve_target_hostname("192.168.1.100") == "box.local"

    def test_localhost_falls_back(self):
        with patch(self._FQDN_PATCH, return_value="box.local"):
            assert resolve_target_hostname("localhost") == "box.local"


# ---------------------------------------------------------------------------
# build_agent_record
# ---------------------------------------------------------------------------


class TestBuildAgentRecord:
    def test_disabled(self):
        args = _make_args(dns_aid_enabled=False)
        assert build_agent_record(args, _make_engine_client()) is None

    def test_missing_library(self):
        args = _make_args()
        with patch.object(dns_aid_mod, "_is_dns_aid_available", return_value=False):
            result = build_agent_record(args, _make_engine_client())
        assert result is None

    def test_no_zone(self, monkeypatch):
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        args = _make_args(dns_aid_zone=None)
        with patch.object(dns_aid_mod, "_is_dns_aid_available", return_value=True):
            result = build_agent_record(args, _make_engine_client())
        assert result is None

    def test_success(self, monkeypatch, fake_dns_aid):
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        monkeypatch.delenv("DNS_AID_NAME", raising=False)

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            record = build_agent_record(_make_args(), _make_engine_client())

        assert record is not None
        expected_name = "meta-llama-llama-3-70b-instruct._agents.example.internal"
        assert record.name == expected_name
        assert record.target == "serve.example.internal"
        assert record.port == 8000
        assert record.params.alpn == ["h2"]
        hints = record.params.hints
        assert hints["model"] == "meta-llama/Llama-3-70b-instruct"
        assert hints["context_len"] == "131072"
        assert hints["quant"] == "fp8"
        assert hints["max_batch"] == "256"
        assert hints["framework"] == "vllm"
        assert hints["api_base"] == "/v1"

    def test_custom_name(self, monkeypatch, fake_dns_aid):
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        monkeypatch.delenv("DNS_AID_NAME", raising=False)

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            record = build_agent_record(
                _make_args(dns_aid_name="my-llama"),
                _make_engine_client(),
            )
        assert record.name == "my-llama._agents.example.internal"

    def test_nonzero_global_rank_skips(self, fake_dns_aid):
        """Any non-zero global rank is skipped, regardless of topology."""
        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=False):
            # TP > 1
            assert (
                build_agent_record(
                    _make_args(),
                    _make_engine_client(tensor_parallel_size=2),
                )
                is None
            )
            # TP = 1 (e.g. PP-only or DP-only)
            assert (
                build_agent_record(
                    _make_args(),
                    _make_engine_client(tensor_parallel_size=1),
                )
                is None
            )

    def test_quant_none(self, monkeypatch, fake_dns_aid):
        """When quantization is None, the hint should be 'none'."""
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        monkeypatch.delenv("DNS_AID_NAME", raising=False)

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            record = build_agent_record(
                _make_args(),
                _make_engine_client(quantization=None),
            )
        assert record.params.hints["quant"] == "none"

    def test_zone_from_env(self, monkeypatch, fake_dns_aid):
        """Zone from DNS_AID_ZONE env var when CLI arg is absent."""
        monkeypatch.setenv("DNS_AID_ZONE", "env.example.com")
        monkeypatch.delenv("DNS_AID_NAME", raising=False)

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            record = build_agent_record(
                _make_args(dns_aid_zone=None),
                _make_engine_client(),
            )
        assert "env.example.com" in record.name


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    @pytest.mark.asyncio
    async def test_calls_register_agent(self, monkeypatch, fake_dns_aid):
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        monkeypatch.delenv("DNS_AID_NAME", raising=False)

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            record = await dns_aid_mod.register(_make_args(), _make_engine_client())

        assert record is not None
        fake_dns_aid.register_agent.assert_called_once_with(record)

    @pytest.mark.asyncio
    async def test_exception_returns_none(self, monkeypatch, fake_dns_aid):
        monkeypatch.delenv("DNS_AID_ZONE", raising=False)
        monkeypatch.delenv("DNS_AID_NAME", raising=False)
        fake_dns_aid.register_agent.side_effect = RuntimeError("DNS timeout")

        with patch.object(dns_aid_mod, "_is_global_rank_zero", return_value=True):
            result = await dns_aid_mod.register(_make_args(), _make_engine_client())

        assert result is None  # server must not crash

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        result = await dns_aid_mod.register(
            _make_args(dns_aid_enabled=False), _make_engine_client()
        )
        assert result is None


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------


class TestDeregister:
    @pytest.mark.asyncio
    async def test_none_is_noop(self):
        await dns_aid_mod.deregister(None)  # must not raise

    @pytest.mark.asyncio
    async def test_calls_deregister_agent(self, fake_dns_aid):
        fake_record = MagicMock()
        fake_record.name = "my-agent._agents.example.internal"

        await dns_aid_mod.deregister(fake_record)

        fake_dns_aid.deregister_agent.assert_called_once_with(fake_record)

    @pytest.mark.asyncio
    async def test_exception_does_not_raise(self, fake_dns_aid):
        fake_dns_aid.deregister_agent.side_effect = RuntimeError("DNS unreachable")
        fake_record = MagicMock()
        fake_record.name = "my-agent._agents.example.internal"

        await dns_aid_mod.deregister(fake_record)  # must not raise
