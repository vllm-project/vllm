"""Interpreter-startup hook: when VLLM_USE_RUST_BLOCK_POOL=1, patch
`vllm.v1.core.block_pool.BlockPool` to use `RustBlockPool` before any
user code — including vLLM's spawned EngineCore subprocesses — imports
the scheduler.

Drop this directory on PYTHONPATH.
"""
import os
import sys

if os.environ.get("VLLM_USE_RUST_BLOCK_POOL") != "1":
    # Fall through to the rest of sitecustomize discovery.
    pass
else:
    try:
        # Register a meta-path finder that intercepts the first import of
        # vllm.v1.core.block_pool and does the swap afterwards.
        import importlib
        import importlib.util

        _REPO_ROOT = os.environ.get(
            "VLLM_RS_REPO_ROOT",
            "/home/yuchao/workspace/tmp/rust-accelerate/vllm",
        )

        class _Hook:
            _patched = False

            def find_spec(self, name, path, target=None):
                if self._patched:
                    return None
                if name != "vllm.v1.core.block_pool":
                    return None
                self._patched = True
                # Let the real import proceed via other finders — we'll patch
                # once it finishes.
                # Find the real spec via the standard mechanism.
                sys.meta_path.remove(self)
                try:
                    spec = importlib.util.find_spec(name)
                finally:
                    # After the block_pool module is actually loaded by the
                    # import machinery we'll run our patch. We schedule it via
                    # a loader wrapper.
                    pass
                if spec is None or spec.loader is None:
                    return None
                original_exec = spec.loader.exec_module

                def exec_and_patch(module):
                    original_exec(module)
                    _swap_block_pool(module)

                spec.loader.exec_module = exec_and_patch  # type: ignore
                return spec

        def _swap_block_pool(bp_module):
            shim_path = os.path.join(
                _REPO_ROOT, "vllm", "v1", "core", "rust_block_pool.py"
            )
            if not os.path.exists(shim_path):
                print(
                    f"[sitecustomize] RustBlockPool disabled — missing {shim_path}",
                    file=sys.stderr,
                )
                return
            spec = importlib.util.spec_from_file_location(
                "vllm.v1.core.rust_block_pool", shim_path
            )
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["vllm.v1.core.rust_block_pool"] = mod
            spec.loader.exec_module(mod)
            bp_module.BlockPool = mod.RustBlockPool
            print(
                f"[sitecustomize pid={os.getpid()}] "
                f"swapped vllm.v1.core.block_pool.BlockPool -> "
                f"{mod.RustBlockPool.__module__}.{mod.RustBlockPool.__name__}",
                file=sys.stderr,
                flush=True,
            )

        sys.meta_path.insert(0, _Hook())
    except Exception as exc:  # pragma: no cover
        print(f"[sitecustomize] RustBlockPool patch failed: {exc!r}", file=sys.stderr)
