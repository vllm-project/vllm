# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that the draft-token hook in post_step() fires whenever the model
was executed, not only when speculative decoding is enabled.

This enables plugins (e.g. dLLM schedulers/workers) to reuse the
spec-decode data path by providing draft token ids from
take_draft_token_ids().

See: https://github.com/vllm-project/vllm/issues/36155
"""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test


class TestPostStepDraftHook:
    """Unit tests for EngineCore.post_step draft-token hook behavior."""

    def _make_engine_core_stub(
        self, use_spec_decode: bool, async_scheduling: bool = False
    ):
        """Create a minimal stub of EngineCore with mocked dependencies."""
        # We can't instantiate EngineCore directly (requires GPU), so we
        # create a simple object with the same attributes post_step uses.
        stub = MagicMock()
        stub.use_spec_decode = use_spec_decode
        stub.async_scheduling = async_scheduling
        stub.model_executor = MagicMock()
        stub.scheduler = MagicMock()
        return stub

    def test_post_step_calls_hook_without_spec_decode(self):
        """post_step should call take_draft_token_ids even when
        use_spec_decode is False, enabling plugin-provided draft tokens."""
        from vllm.v1.engine.core import EngineCore

        stub = self._make_engine_core_stub(use_spec_decode=False)
        draft_ids = MagicMock()
        stub.model_executor.take_draft_token_ids.return_value = draft_ids

        # Call post_step with model_executed=True
        EngineCore.post_step(stub, model_executed=True)

        # The hook should be called even without spec decode
        stub.model_executor.take_draft_token_ids.assert_called_once()
        stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)

    def test_post_step_noop_when_draft_ids_none(self):
        """post_step should not call update_draft_token_ids when
        take_draft_token_ids returns None (default worker behavior)."""
        from vllm.v1.engine.core import EngineCore

        stub = self._make_engine_core_stub(use_spec_decode=False)
        stub.model_executor.take_draft_token_ids.return_value = None

        EngineCore.post_step(stub, model_executed=True)

        stub.model_executor.take_draft_token_ids.assert_called_once()
        stub.scheduler.update_draft_token_ids.assert_not_called()

    def test_post_step_noop_when_model_not_executed(self):
        """post_step should not call the hook when model was not executed."""
        from vllm.v1.engine.core import EngineCore

        stub = self._make_engine_core_stub(use_spec_decode=False)

        EngineCore.post_step(stub, model_executed=False)

        stub.model_executor.take_draft_token_ids.assert_not_called()

    def test_post_step_noop_with_async_scheduling(self):
        """post_step should not call the hook when async scheduling is on,
        because draft tokens are updated in the worker process."""
        from vllm.v1.engine.core import EngineCore

        stub = self._make_engine_core_stub(use_spec_decode=False, async_scheduling=True)

        EngineCore.post_step(stub, model_executed=True)

        stub.model_executor.take_draft_token_ids.assert_not_called()

    def test_post_step_still_works_with_spec_decode(self):
        """post_step should still work correctly when spec decode IS enabled
        (backward compatibility)."""
        from vllm.v1.engine.core import EngineCore

        stub = self._make_engine_core_stub(use_spec_decode=True)
        draft_ids = MagicMock()
        stub.model_executor.take_draft_token_ids.return_value = draft_ids

        EngineCore.post_step(stub, model_executed=True)

        stub.model_executor.take_draft_token_ids.assert_called_once()
        stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)
