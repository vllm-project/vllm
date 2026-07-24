# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
    get_backend_config,
    register_backend,
)


class CustomAttentionImpl(AttentionImpl):
    """Mock custom attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomAttentionBackend(AttentionBackend):
    """Mock custom attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return CustomAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


class CustomMambaAttentionImpl(AttentionImpl):
    """Mock custom mamba attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomMambaAttentionBackend(AttentionBackend):
    """Mock custom mamba attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM_MAMBA"

    @staticmethod
    def get_impl_cls():
        return CustomMambaAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


def test_custom_is_not_alias_of_any_backend():
    # Get all members of AttentionBackendEnum
    all_backends = list(AttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is AttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(AttentionBackendEnum.CUSTOM.value)}\n"
        f"This happens when CUSTOM has the same value as another backend.\n"
        f"When you register to CUSTOM, you're actually registering to {aliases[0]}!\n"
        f"All backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    # Verify CUSTOM has its own unique identity
    assert AttentionBackendEnum.CUSTOM.name == "CUSTOM", (
        f"CUSTOM.name should be 'CUSTOM', but got '{AttentionBackendEnum.CUSTOM.name}'"
    )


def test_register_custom_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
        is_mamba=False,
    )

    # Check that CUSTOM backend is registered
    assert AttentionBackendEnum.CUSTOM.is_overridden(), (
        "CUSTOM should be overridden after registration"
    )

    # Get the registered class path
    class_path = AttentionBackendEnum.CUSTOM.get_path()
    assert class_path == "tests.test_attention_backend_registry.CustomAttentionBackend"

    # Get the backend class
    backend_cls = AttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"
    assert backend_cls.get_impl_cls() == CustomAttentionImpl


def test_mamba_custom_is_not_alias_of_any_backend():
    # Get all mamba backends
    all_backends = list(MambaAttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MambaAttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! MambaAttentionBackendEnum.CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MambaAttentionBackendEnum.CUSTOM.value)}\n"
        f"All mamba backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )


def test_register_custom_mamba_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=MambaAttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomMambaAttentionBackend",
        is_mamba=True,
    )

    # Check that the backend is registered
    assert MambaAttentionBackendEnum.CUSTOM.is_overridden()

    # Get the registered class path
    class_path = MambaAttentionBackendEnum.CUSTOM.get_path()
    assert (
        class_path
        == "tests.test_attention_backend_registry.CustomMambaAttentionBackend"
    )

    # Get the backend class
    backend_cls = MambaAttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM_MAMBA"
    assert backend_cls.get_impl_cls() == CustomMambaAttentionImpl


def test_register_backend_config_with_class_path():
    """Test passing backend_config via register_backend() with class_path."""
    config = {"log_level": "debug", "sample_rate": 0.1}

    # Clean up any prior state
    AttentionBackendEnum.CUSTOM.clear_override()
    AttentionBackendEnum.CUSTOM.clear_config()

    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
        backend_config=config,
    )

    # Config should be retrievable via the enum method
    assert AttentionBackendEnum.CUSTOM.get_config() == config
    # Config should also be retrievable via the module-level function
    assert get_backend_config(AttentionBackendEnum.CUSTOM) == config

    # Clean up
    AttentionBackendEnum.CUSTOM.clear_config()
    assert AttentionBackendEnum.CUSTOM.get_config() is None


def test_register_backend_config_with_decorator():
    """Test passing backend_config via the @register_backend decorator."""
    config = {"enable_tracing": True}

    # Clean up any prior state
    AttentionBackendEnum.CUSTOM.clear_override()
    AttentionBackendEnum.CUSTOM.clear_config()

    @register_backend(AttentionBackendEnum.CUSTOM, backend_config=config)
    class _DecoratedBackend(CustomAttentionBackend):
        pass

    assert AttentionBackendEnum.CUSTOM.get_config() == config
    assert get_backend_config(AttentionBackendEnum.CUSTOM) == config

    # Clean up
    AttentionBackendEnum.CUSTOM.clear_config()
    AttentionBackendEnum.CUSTOM.clear_override()


def test_register_mamba_backend_config():
    """Test passing backend_config for a MambaAttentionBackendEnum."""
    config = {"conv_width": 4}

    MambaAttentionBackendEnum.CUSTOM.clear_config()

    register_backend(
        backend=MambaAttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomMambaAttentionBackend",
        is_mamba=True,
        backend_config=config,
    )

    assert MambaAttentionBackendEnum.CUSTOM.get_config() == config
    assert get_backend_config(MambaAttentionBackendEnum.CUSTOM) == config

    # Clean up
    MambaAttentionBackendEnum.CUSTOM.clear_config()


def test_backend_config_defaults_to_none():
    """Test that config is None when not explicitly set."""
    AttentionBackendEnum.CUSTOM.clear_config()

    # Register without backend_config
    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
    )

    assert AttentionBackendEnum.CUSTOM.get_config() is None
    assert get_backend_config(AttentionBackendEnum.CUSTOM) is None


def test_backend_config_clear():
    """Test that clear_config removes the stored config."""
    config = {"key": "value"}

    AttentionBackendEnum.CUSTOM.clear_config()

    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
        backend_config=config,
    )
    assert AttentionBackendEnum.CUSTOM.get_config() == config

    AttentionBackendEnum.CUSTOM.clear_config()
    assert AttentionBackendEnum.CUSTOM.get_config() is None
