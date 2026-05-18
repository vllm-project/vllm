# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry


class DummyLoRAResolver(LoRAResolver):
    """A dummy LoRA resolver for testing."""

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        if lora_name == "test_lora":
            return LoRARequest(
                lora_name=lora_name,
                lora_path=f"/dummy/path/{base_model_name}/{lora_name}",
                lora_int_id=abs(hash(lora_name)),
            )
        return None


def test_resolver_registry_registration():
    """Test basic resolver registration functionality."""
    registry = LoRAResolverRegistry
    resolver = DummyLoRAResolver()

    # Register a new resolver
    registry.register_resolver("dummy", resolver)
    assert "dummy" in registry.get_supported_resolvers()

    # Get registered resolver
    retrieved_resolver = registry.get_resolver("dummy")
    assert retrieved_resolver is resolver


def test_resolver_registry_duplicate_registration():
    """Test registering a resolver with an existing name."""
    registry = LoRAResolverRegistry
    resolver1 = DummyLoRAResolver()
    resolver2 = DummyLoRAResolver()

    registry.register_resolver("dummy", resolver1)
    registry.register_resolver("dummy", resolver2)

    assert registry.get_resolver("dummy") is resolver2


def test_resolver_registry_unknown_resolver():
    """Test getting a non-existent resolver."""
    registry = LoRAResolverRegistry

    with pytest.raises(KeyError, match="not found"):
        registry.get_resolver("unknown_resolver")


@pytest.mark.asyncio
async def test_dummy_resolver_resolve():
    """Test the dummy resolver's resolve functionality."""
    dummy_resolver = DummyLoRAResolver()
    base_model_name = "base_model_test"
    lora_name = "test_lora"

    # Test successful resolution
    result = await dummy_resolver.resolve_lora(base_model_name, lora_name)
    assert isinstance(result, LoRARequest)
    assert result.lora_name == lora_name
    assert result.lora_path == f"/dummy/path/{base_model_name}/{lora_name}"

    # Test failed resolution
    result = await dummy_resolver.resolve_lora(base_model_name, "nonexistent_lora")
    assert result is None
