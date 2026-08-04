# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import subprocess
import sys
import textwrap

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_import_script(script: str) -> None:
    """Run an import check in a fresh Python subprocess.

    Args:
        script: Python source code to execute with ``python -c``.

    Raises:
        AssertionError: If the subprocess exits with a non-zero status.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_memory_allocators_package_import_is_lazy() -> None:
    """Importing the allocator package does not eagerly import submodules."""
    _run_import_script(
        """
        import sys

        import lmcache.v1.memory_allocators as allocators

        allocator_modules = {
            "ad_hoc_memory_allocator",
            "buffer_allocator",
            "cu_file_memory_allocator",
            "devdax_memory_allocator",
            "gpu_memory_allocator",
            "hip_file_memory_allocator",
            "host_memory_allocator",
            "lazy_memory_allocator",
            "mixed_memory_allocator",
            "paged_cpu_gpu_memory_allocator",
            "paged_tensor_memory_allocator",
            "pin_memory_allocator",
            "tensor_memory_allocator",
        }
        loaded = sorted(
            name
            for name in sys.modules
            if name.startswith("lmcache.v1.memory_allocators.")
            and name.rsplit(".", 1)[-1] in allocator_modules
        )
        assert loaded == [], loaded
        assert "LazyMemoryAllocator" in allocators.__all__
        assert "TensorMemoryAllocator" in allocators.__all__
        """
    )


def test_allocator_submodule_first_import_keeps_core_surface() -> None:
    """Importing concrete allocator modules first does not create a cycle."""
    _run_import_script(
        """
        from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
            PagedAddressManager,
            PagedTensorMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            MemoryAllocatorInterface,
            MemoryFormat,
        )

        assert PagedAddressManager.__name__ == "PagedAddressManager"
        assert PagedTensorMemoryAllocator.__name__ == "PagedTensorMemoryAllocator"
        assert TensorMemoryAllocator.__name__ == "TensorMemoryAllocator"
        assert MemoryAllocatorInterface.__name__ == "MemoryAllocatorInterface"
        assert MemoryFormat.KV_2LTD.name == "KV_2LTD"
        """
    )


def test_multiple_allocator_submodule_imports_keep_core_surface() -> None:
    """Concrete allocator imports do not disturb core memory_management imports."""
    _run_import_script(
        """
        from lmcache.v1.memory_allocators.devdax_memory_allocator import (
            DevDaxMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            MemoryObj,
            TensorMemoryObj,
        )

        assert DevDaxMemoryAllocator.__name__ == "DevDaxMemoryAllocator"
        assert TensorMemoryAllocator.__name__ == "TensorMemoryAllocator"
        assert MemoryObj.__name__ == "MemoryObj"
        assert TensorMemoryObj.__name__ == "TensorMemoryObj"
        """
    )


def test_lazy_allocator_package_and_submodule_paths_match() -> None:
    """The lazy allocator package export matches its concrete submodule."""
    _run_import_script(
        """
        import lmcache.v1.memory_allocators as allocators

        from lmcache.v1.memory_allocators.lazy_memory_allocator import (
            LazyMemoryAllocator,
        )

        assert LazyMemoryAllocator is allocators.LazyMemoryAllocator
        """
    )


def test_memory_management_all_keeps_only_core_names() -> None:
    """The memory_management import surface lists only core memory types."""
    _run_import_script(
        """
        import lmcache.v1.memory_management as memory_management

        allocator_names = {
            "AdHocMemoryAllocator",
            "BufferAllocator",
            "CuFileMemoryAllocator",
            "DevDaxMemoryAllocator",
            "GPUMemoryAllocator",
            "HipFileMemoryAllocator",
            "HostMemoryAllocator",
            "MixedMemoryAllocator",
            "PagedAddressManager",
            "PagedCpuGpuMemoryAllocator",
            "PagedTensorMemoryAllocator",
            "PinMemoryAllocator",
            "TensorMemoryAllocator",
        }
        expected = {
            "AddressManager",
            "BytesBufferMemoryObj",
            "FreeBlock",
            "GDSMemoryObject",
            "MemoryAllocatorInterface",
            "MemoryFormat",
            "MemoryObj",
            "MemoryObjMetadata",
            "TensorMemoryObj",
            "torch_device_type",
        }
        missing = expected - set(memory_management.__all__)
        assert missing == set(), missing
        assert allocator_names.isdisjoint(memory_management.__all__)

        assert not hasattr(memory_management, "TensorMemoryAllocator")
        """
    )
