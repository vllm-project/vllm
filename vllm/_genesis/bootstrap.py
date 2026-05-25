"""
[Genesis] Bootstrap — apply PN61, PN62, P83 at import time.
Call this module from vllm's startup or let the _genesis dispatcher
trigger it automatically.
"""
import logging

log = logging.getLogger("genesis.bootstrap")


def bootstrap_all() -> list[tuple[str, str]]:
    """Apply all three remaining patches and return status for each."""
    results = []

    # PN61: qwen3_vl NVFP4 loader guard
    from vllm._genesis.pn61_guard import apply_pn61
    status, msg = apply_pn61()
    results.append(("PN61", status, msg))
    log.info("PN61: %s — %s", status, msg)

    # PN62: GPUModelRunner._dummy_run ViT scratch skip
    from vllm._genesis.pn62_guard import apply_pn62
    status, msg = apply_pn62()
    results.append(("PN62", status, msg))
    log.info("PN62: %s — %s", status, msg)

    # P83: text-patch already applied directly to source (no wrapper needed)
    results.append(("P83", "applied", "P83 text-patch already in single_type_kv_cache_manager.py"))
    log.info("P83: applied — text-patch in source")

    return results


# Auto-apply on import
_bootstrap_results = bootstrap_all()
