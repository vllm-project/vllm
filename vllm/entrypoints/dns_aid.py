"""DNS-AID SVCB endpoint registration for vLLM.

When enabled, registers a DNS-AID SVCB ServiceMode record on startup encoding
model capabilities (model name, context window, quantization, max batch size,
transport). Agents can discover vLLM endpoints via a single DNS lookup.

Optional dependency: dns-aid (pip install vllm[dns-aid])
"""

import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from dns_aid import AgentRecord

    from vllm.engine.protocol import EngineClient

logger = logging.getLogger(__name__)

# Maximum length of a single DNS label (RFC 1035).
_MAX_DNS_LABEL_LEN = 63


def _is_dns_aid_available() -> bool:
    """Return True if the dns-aid library is importable."""
    return "dns_aid" in __import__("sys").modules or _try_import_dns_aid()


def _try_import_dns_aid() -> bool:
    try:
        import dns_aid  # noqa: F401

        return True
    except ImportError:
        return False


def slugify(model_name: str) -> str:
    """Convert a model name to a DNS-safe label.

    If the resulting slug exceeds the 63-octet DNS label limit, it is
    truncated and a short hash suffix is appended to preserve uniqueness.

    Examples:
        "meta-llama/Llama-3-70b-instruct" -> "meta-llama-llama-3-70b-instruct"
        "Qwen/Qwen3-0.6B"                 -> "qwen-qwen3-0-6b"
    """
    slug = model_name.lower()
    slug = re.sub(r"[/_.\s]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")

    if len(slug) > _MAX_DNS_LABEL_LEN:
        # 8-char hex hash suffix + separator = 9 chars reserved
        digest = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        slug = slug[: _MAX_DNS_LABEL_LEN - 9] + "-" + digest

    return slug


def resolve_target_hostname(host_arg: str | None) -> str:
    """Return the best FQDN to advertise in the SVCB record.

    Uses *host_arg* directly if it looks like a resolvable FQDN (contains a
    dot, is not an IP address).  Falls back to ``socket.getfqdn()`` otherwise.
    """
    if host_arg:
        try:
            ipaddress.ip_address(host_arg)
        except ValueError:
            # Not an IP — treat as FQDN if it has a dot
            if "." in host_arg:
                return host_arg
    return socket.getfqdn()


def _is_global_rank_zero() -> bool:
    """Return True if this process is the global rank-0 worker.

    In multi-process setups (TP, PP, DP) only global rank 0 should register
    the DNS record so that exactly one record exists per deployment.  When
    ``torch.distributed`` is not initialised (single-process), returns True.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True  # single-process: always register


def build_agent_record(
    args: "Namespace",
    engine_client: "EngineClient",
) -> "AgentRecord | None":
    """Build a dns_aid.AgentRecord from server args and engine state.

    Returns None (with a warning) if any guard condition is not met:
    - dns_aid_enabled is False
    - dns-aid library not installed
    - no DNS zone configured
    - tensor-parallel rank != 0
    """
    if not getattr(args, "dns_aid_enabled", False):
        return None

    if not _is_dns_aid_available():
        logger.warning(
            "DNS-AID: --dns-aid-enabled is set but the 'dns-aid' library is "
            "not installed. Run: pip install 'vllm[dns-aid]'. Skipping."
        )
        return None

    zone = getattr(args, "dns_aid_zone", None) or os.environ.get("DNS_AID_ZONE")
    if not zone:
        logger.warning(
            "DNS-AID: --dns-aid-enabled is set but no zone is configured. "
            "Set --dns-aid-zone or DNS_AID_ZONE. Skipping."
        )
        return None

    # Only global rank 0 registers so that exactly one record exists per
    # deployment, regardless of TP/PP/DP topology.
    if not _is_global_rank_zero():
        logger.debug("DNS-AID: skipping registration on non-zero global rank.")
        return None

    import dns_aid  # already confirmed importable

    model_name = engine_client.model_config.model
    agent_name = (
        getattr(args, "dns_aid_name", None)
        or os.environ.get("DNS_AID_NAME")
        or slugify(model_name)
    )

    target = resolve_target_hostname(getattr(args, "host", None))
    port = getattr(args, "port", 8000)

    # Collect model capability hints
    try:
        context_len = str(engine_client.model_config.max_model_len)
    except Exception:
        context_len = "unknown"

    try:
        quant = engine_client.model_config.quantization or "none"
        if not isinstance(quant, str):
            quant = str(quant)
    except Exception:
        quant = "none"

    try:
        max_batch = str(engine_client.vllm_config.scheduler_config.max_num_seqs)
    except Exception:
        max_batch = "unknown"

    record = dns_aid.AgentRecord(
        name=f"{agent_name}._agents.{zone}",
        target=target,
        port=port,
        params=dns_aid.SvcParams(
            alpn=["h2"],
            hints={
                "model": model_name,
                "context_len": context_len,
                "quant": quant,
                "max_batch": max_batch,
                "framework": "vllm",
                "api_base": "/v1",
            },
        ),
        # Short TTL so that clients discover replacement endpoints quickly
        # after a rolling restart or scale-down.  The dns-aid library handles
        # periodic refresh; vLLM itself does not need to re-register.
        ttl=60,
    )
    return record


async def register(
    args: "Namespace",
    engine_client: "EngineClient",
) -> "AgentRecord | None":
    """Register the vLLM endpoint in DNS-AID.

    Returns the AgentRecord on success (needed for deregistration), or None
    if registration was skipped or failed. Never raises -- a DNS failure must
    not prevent the server from starting.
    """
    record = build_agent_record(args, engine_client)
    if record is None:
        return None

    import dns_aid

    try:
        await asyncio.to_thread(dns_aid.register_agent, record)
        logger.info(
            "DNS-AID: registered agent '%s' at target '%s:%s'.",
            record.name,
            record.target,
            record.port,
        )
        return record
    except Exception as exc:
        logger.warning("DNS-AID: registration failed: %s", exc, exc_info=True)
        return None


async def deregister(record: "AgentRecord | None") -> None:
    """Deregister the vLLM endpoint from DNS-AID.

    Safe to call with None (no-op). Never raises.
    """
    if record is None:
        return

    import dns_aid

    try:
        await asyncio.to_thread(dns_aid.deregister_agent, record)
        logger.info("DNS-AID: deregistered agent '%s'.", record.name)
    except Exception as exc:
        logger.warning("DNS-AID: deregistration failed: %s", exc, exc_info=True)
