# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-runner auto-merge state: tracks which adapter is currently merged."""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from .merge import BF16GoldenCache, bf16_restore_base, merge_lora_into_base

logger = init_logger(__name__)


@dataclass
class AutoMergeState:
    verbose: bool = True
    validate_dtypes: bool = True
    golden_device: str = "cpu"  # "cpu", "gpu", or "off"

    merged_lora_name: str | None = None
    merged_adapter_id: int | None = None
    merged_at_ms: int | None = None

    merge_count: int = 0
    unmerge_count: int = 0
    last_error: str | None = None

    _bf16_cache: BF16GoldenCache = field(init=False)

    def __post_init__(self):
        self._bf16_cache = BF16GoldenCache(device=self.golden_device)

    def _get_model_manager(self, runner: Any) -> Any | None:
        lora_manager = getattr(runner, "lora_manager", None)
        if lora_manager is None:
            return None
        mm = getattr(lora_manager, "_adapter_manager", None)
        if mm is None:
            mm = getattr(lora_manager, "model_manager", None)
        return mm

    def _get_active_adapter_id(self, runner: Any) -> int | None:
        mm = self._get_model_manager(runner)
        if mm is None:
            return None
        lora_index_to_id = getattr(mm, "lora_index_to_id", None)
        if not lora_index_to_id:
            return None
        return lora_index_to_id[0]

    def _get_lora_model_by_id(self, runner: Any, adapter_id: int) -> Any | None:
        mm = self._get_model_manager(runner)
        if mm is None:
            return None
        get_adapter = getattr(mm, "get_adapter", None)
        if get_adapter is None:
            return None
        return get_adapter(adapter_id)

    def unmerge_if_needed(self, runner: Any) -> None:
        if self.merged_adapter_id is None:
            return

        mm = self._get_model_manager(runner)
        if mm is None:
            self.last_error = "no model_manager during unmerge"
            return

        lora_model = self._get_lora_model_by_id(runner, self.merged_adapter_id)
        if lora_model is not None:
            lora_names = set(getattr(lora_model, "loras", {}).keys())
        elif self._bf16_cache.initialized:
            lora_names = set(self._bf16_cache._cache.keys())
        else:
            logger.warning(
                "automerge: no lora_model and no golden cache for unmerge of id=%s",
                self.merged_adapter_id,
            )
            self.merged_adapter_id = None
            self.merged_lora_name = None
            return

        res = bf16_restore_base(
            model_manager=mm,
            golden_cache=self._bf16_cache,
            lora_module_names=lora_names,
            lora_model=lora_model,
        )

        if res.ok:
            self.unmerge_count += 1
            logger.info(
                "automerge: restored base for lora_name=%s (layers=%d, skipped=%d)",
                self.merged_lora_name,
                res.merged_modules,
                res.skipped_modules,
            )
        else:
            self.last_error = f"bf16 restore failed: {res.reason}"
            logger.error("automerge: %s", self.last_error)

        self.merged_adapter_id = None
        self.merged_lora_name = None
        self.merged_at_ms = None

    def merge_active(
        self,
        runner: Any,
        desired_lora_name: str,
        adapter_id: int | None = None,
    ) -> bool:
        mm = self._get_model_manager(runner)
        if mm is None:
            self.last_error = "no model_manager during merge"
            return False

        if adapter_id is None:
            adapter_id = self._get_active_adapter_id(runner)
        if adapter_id is None:
            self.last_error = "no active adapter in slot 0"
            return False

        lora_model = self._get_lora_model_by_id(runner, adapter_id)
        if lora_model is None:
            self.last_error = f"adapter_id {adapter_id} not found in manager"
            return False

        res = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=self._bf16_cache,
            validate_dtypes=self.validate_dtypes,
        )

        if not res.ok:
            self.last_error = f"merge failed: {res.reason}"
            logger.error("automerge: %s", self.last_error)
            # If some layers were partially merged, restore them
            if res.merged_modules > 0:
                lora_names = set(getattr(lora_model, "loras", {}).keys())
                bf16_restore_base(
                    model_manager=mm,
                    golden_cache=self._bf16_cache,
                    lora_module_names=lora_names,
                    lora_model=lora_model,
                )
                logger.info("automerge: cleaned up partial merge")
            return False

        self.merged_lora_name = desired_lora_name
        self.merged_adapter_id = adapter_id
        self.merged_at_ms = int(time.time() * 1000)
        self.merge_count += 1
        logger.info(
            "automerge: merged lora_name=%s adapter_id=%d (layers=%d, skipped=%d)",
            desired_lora_name,
            adapter_id,
            res.merged_modules,
            res.skipped_modules,
        )
        return True


_runner_state: weakref.WeakKeyDictionary[Any, AutoMergeState] = (
    weakref.WeakKeyDictionary()
)


def get_state(runner: Any, golden_device: str = "cpu") -> AutoMergeState:
    st = _runner_state.get(runner)
    if st is None:
        st = AutoMergeState(golden_device=golden_device)
        _runner_state[runner] = st
    return st
