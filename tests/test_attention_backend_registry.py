# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.selector import AttentionSelectorConfig


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


class StringDecoratorAttentionBackend(AttentionBackend):
    """Mock backend defined at module level so get_class() can resolve
    the auto-generated class path (local classes are not importable)."""

    @staticmethod
    def get_name():
        return "DECORATED"

    @staticmethod
    def get_impl_cls():
        return CustomAttentionImpl

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        return None


class StringDecoratorMambaBackend(AttentionBackend):
    """Mock mamba backend defined at module level so get_class() can
    resolve the auto-generated class path."""

    @staticmethod
    def get_name():
        return "DECORATED"

    @staticmethod
    def get_impl_cls():
        return CustomMambaAttentionImpl

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        return None


def test_custom_attention_backend_requires_dcp_opt_in():
    """A new implementation remains usable without implicitly opting into DCP."""
    config = AttentionSelectorConfig(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
    )
    kwargs = dict(device_capability=DeviceCapability(9, 0), **config._asdict())
    assert CustomAttentionBackend.validate_configuration(**kwargs) == []

    kwargs["use_dcp"] = True
    assert CustomAttentionBackend.validate_configuration(**kwargs) == [
        "DCP not supported"
    ]


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


@pytest.fixture(params=[AttentionBackendEnum, MambaAttentionBackendEnum])
def _enum_cls(request):
    return request.param


def _remove_dynamic_member(enum_cls, member):
    value = member.value
    name = member.name
    enum_cls._member_map_.pop(name, None)
    enum_cls._member_names_.remove(name)
    delattr(enum_cls, name)
    if enum_cls._value2member_map_.get(value) is member:
        del enum_cls._value2member_map_[value]


def test_register_dynamic_enum_member(_enum_cls):
    backend_path = {
        AttentionBackendEnum: (
            "tests.test_attention_backend_registry.CustomAttentionBackend"
        ),
        MambaAttentionBackendEnum: (
            "tests.test_attention_backend_registry.CustomMambaAttentionBackend"
        ),
    }[_enum_cls]

    member = _enum_cls.register("DYNAMIC_TEST", backend_path)
    assert member.name == "DYNAMIC_TEST"
    assert member.value == backend_path
    assert member is _enum_cls.DYNAMIC_TEST
    assert _enum_cls["DYNAMIC_TEST"] is member
    _remove_dynamic_member(_enum_cls, member)


def test_register_dynamic_enum_member_duplicate_raises(_enum_cls):
    _enum_cls.register("DUP_TEST", "some.module.Class")
    with pytest.raises(ValueError, match="already exists"):
        _enum_cls.register("DUP_TEST", "other.module.OtherClass")
    _remove_dynamic_member(_enum_cls, _enum_cls.DUP_TEST)


def test_register_dynamic_enum_member_reserved_name_raises(_enum_cls):
    with pytest.raises(ValueError, match="Invalid or reserved backend name"):
        _enum_cls.register("get_path", "some.module.Class")


@pytest.fixture(
    params=[
        pytest.param((AttentionBackendEnum, False), id="attention"),
        pytest.param((MambaAttentionBackendEnum, True), id="mamba"),
    ]
)
def _enum_is_mamba(request):
    return request.param


def _backend_path(enum_cls):
    if enum_cls is AttentionBackendEnum:
        return "tests.test_attention_backend_registry.CustomAttentionBackend"
    return "tests.test_attention_backend_registry.CustomMambaAttentionBackend"


def test_register_backend_with_string_name_direct(_enum_is_mamba):
    enum_cls, is_mamba = _enum_is_mamba
    path = _backend_path(enum_cls)
    register_backend("STRING_DIRECT", path, is_mamba=is_mamba)

    member = enum_cls.STRING_DIRECT
    assert member.is_overridden()
    assert member.get_path() == path
    expected = "CUSTOM" if not is_mamba else "CUSTOM_MAMBA"
    assert member.get_class().get_name() == expected
    member.clear_override()
    _remove_dynamic_member(enum_cls, member)


def test_register_backend_with_string_name_decorator(_enum_is_mamba):
    enum_cls, is_mamba = _enum_is_mamba
    # The decorated class must be module-level: the decorator derives a
    # class path from it, and get_class() re-imports that path. A class
    # defined inside this test would have a non-importable <locals> path.
    decorated_cls = (
        StringDecoratorMambaBackend if is_mamba else StringDecoratorAttentionBackend
    )

    result = register_backend("STRING_DECORATOR", is_mamba=is_mamba)(decorated_cls)
    assert result is decorated_cls

    member = enum_cls.STRING_DECORATOR
    assert member.is_overridden()
    assert member.get_path() == (
        f"{decorated_cls.__module__}.{decorated_cls.__qualname__}"
    )
    assert member.get_class() is decorated_cls
    assert member.get_class().get_name() == "DECORATED"
    member.clear_override()
    _remove_dynamic_member(enum_cls, member)
