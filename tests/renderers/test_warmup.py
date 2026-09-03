# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BaseRenderer.warmup MM-warmup behavior.

These tests exercise:
  - Zero-limit modalities are filtered from mm_counts passed to
    get_dummy_processor_inputs (e.g. --limit-mm-per-prompt image=0 ...)
  - MM warmup is skipped entirely when mm_processor is None
  - The multimodal warmup is launched as a task on the single-worker
    _mm_executor to overlap engine-core init (future lifecycle, join by
    warmup/reset/shutdown, no double-run). Routing it through the same
    executor that serves _process_multimodal keeps the numba workqueue
    parallel region single-threaded, avoiding the "Concurrent access has been
    detected" fatal abort when a request arrives during warmup.
  - That overlap only exists for the multiprocess clients: the in-process
    EngineCore is built synchronously, so InprocClient rejects a renderer
    (assert) and warmup stays inside renderer.warmup().

No model weights are required: warmup() is called directly on a MagicMock
that acts as the renderer instance.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import ChatParams


def _make_renderer_mock(mm_limits: dict[str, int]) -> MagicMock:
    """Return a MagicMock that quacks like a BaseRenderer instance.

    render_chat is mocked to raise ChatTemplateResolutionError so the chat
    warmup block is skipped cleanly, keeping the test focused on MM warmup.
    """
    from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

    renderer = MagicMock()

    # chat warmup: make render_chat raise so we skip past it cleanly
    renderer.render_chat.side_effect = ChatTemplateResolutionError("no template")

    # MM processor with configurable limits
    mm_processor = MagicMock()
    mm_processor.info.allowed_mm_limits = mm_limits
    renderer.mm_processor = mm_processor
    renderer._readonly_mm_processor = None
    renderer._warmup_mm_processor = BaseRenderer._warmup_mm_processor.__get__(
        renderer, BaseRenderer
    )
    renderer._clear_processor_cache = BaseRenderer._clear_processor_cache
    renderer.warmup_mm = BaseRenderer.warmup_mm.__get__(renderer, BaseRenderer)
    renderer.start_mm_warmup_in_background = (
        BaseRenderer.start_mm_warmup_in_background.__get__(renderer, BaseRenderer)
    )
    renderer._join_mm_warmup = BaseRenderer._join_mm_warmup.__get__(
        renderer, BaseRenderer
    )
    renderer.shutdown = BaseRenderer.shutdown.__get__(renderer, BaseRenderer)
    # No background warmup launched by default; warmup() takes the inline path.
    renderer._mm_warmup_future = None
    # MM warmup has not run yet; warmup_mm must actually execute on the mock.
    renderer._mm_warmup_done = False
    renderer.clear_mm_cache = MagicMock()
    renderer.model_config.max_model_len = 128
    renderer.model_config.get_multimodal_config.return_value.limit_per_prompt = {}

    return renderer


class TestMmWarmupZeroLimitFiltering:
    """Zero-limit modalities must be excluded from mm_counts."""

    def test_zero_limit_modality_excluded_from_mm_counts(self):
        """A modality with limit=0 must not appear in mm_counts."""
        renderer = _make_renderer_mock({"image": 1, "video": 0})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert "video" not in kwargs["mm_counts"]
        assert kwargs["mm_counts"]["image"] == 1

    def test_all_zero_limits_passes_empty_mm_counts(self):
        """When all limits are 0, mm_counts must be empty."""
        renderer = _make_renderer_mock({"image": 0, "video": 0})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert kwargs["mm_counts"] == {}

    def test_positive_limits_all_included_in_mm_counts(self):
        """All modalities with limit > 0 must be present in mm_counts."""
        renderer = _make_renderer_mock({"image": 2, "video": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert kwargs["mm_counts"] == {"image": 1, "video": 1}


class TestMmWarmupRunsNormally:
    # MM warmup must run when mm_processor is set and limits > 0; the chat
    # template warmup must run alongside it.

    def test_processor_apply_called(self):
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_called_once()

    def test_mm_cache_cleared_after_warmup(self):
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.clear_mm_cache.assert_called_once()

    def test_render_chat_called_with_warmup_message(self):
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.render_chat.assert_called_once()


class TestMmWarmupSkippedWhenNoProcessor:
    """MM warmup must be skipped when mm_processor is None (text-only model)."""

    def test_no_warmup_without_processor(self):
        renderer = _make_renderer_mock({})
        renderer.mm_processor = None  # override to None

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.model_config.get_multimodal_config.assert_not_called()


class TestReadonlyMmWarmup:
    """Readonly MM processor warmup must mirror the render path behavior."""

    def test_readonly_processor_apply_called_and_cache_cleared(self):
        renderer = _make_renderer_mock({"image": 1})
        readonly_mm_processor = MagicMock()
        readonly_mm_processor.info.allowed_mm_limits = {"image": 1}
        renderer._readonly_mm_processor = readonly_mm_processor

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        readonly_mm_processor.apply.assert_called_once()
        readonly_mm_processor.cache.clear_cache.assert_called_once()


class TestWarmupFaultIsolation:
    # A failure during a multimodal processor warmup is caught so it does not
    # abort the remaining warmup steps; warmup itself must not raise.

    def test_chat_failure_does_not_abort_mm_warmup(self):
        renderer = _make_renderer_mock({"image": 1})
        renderer.render_chat.side_effect = RuntimeError("chat boom")

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())  # must not raise

        renderer.mm_processor.apply.assert_called_once()

    def test_mm_failure_does_not_abort_readonly_warmup(self):
        renderer = _make_renderer_mock({"image": 1})
        readonly_mm_processor = MagicMock()
        readonly_mm_processor.info.allowed_mm_limits = {"image": 1}
        renderer._readonly_mm_processor = readonly_mm_processor
        # main processor warmup blows up before apply()
        renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs.side_effect = (
            RuntimeError("mm boom")
        )

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())  # must not raise

        readonly_mm_processor.apply.assert_called_once()
        # cache is still cleared in the failed task's finally
        renderer.clear_mm_cache.assert_called_once()


class TestBackgroundMmWarmup:
    # The multimodal warmup is launched as a task on the single-worker
    # _mm_executor so it overlaps engine-core init (fork) while staying
    # serialized with the serving path (_process_multimodal runs on the same
    # executor). warmup()/reset_mm_cache/shutdown join it. This also fixes the
    # numba workqueue "Concurrent access has been detected" abort: warmup and a
    # concurrent serving request can no longer both enter the numba parallel
    # region at once.

    def _make_bg_renderer(self, mm_limits: dict[str, int]):
        # A renderer mock with a real single-worker executor so submit()
        # returns a real Future and warmup_mm actually runs (in the worker).
        renderer = _make_renderer_mock(mm_limits)
        renderer._mm_executor = ThreadPoolExecutor(max_workers=1)
        return renderer

    def test_start_submits_future_to_mm_executor(self):
        renderer = self._make_bg_renderer({"image": 1})
        try:
            with patch("vllm.multimodal.processing.TimingContext", autospec=True):
                # Spy on the real executor's submit to confirm the warmup is
                # dispatched through _mm_executor (not a separate Thread).
                with patch.object(
                    renderer._mm_executor, "submit", wraps=renderer._mm_executor.submit
                ) as spy:
                    renderer.start_mm_warmup_in_background()
                    assert isinstance(renderer._mm_warmup_future, Future)
                    assert spy.called
                renderer._join_mm_warmup()
            assert renderer._mm_warmup_future is None
            renderer.mm_processor.apply.assert_called_once()
        finally:
            renderer._mm_executor.shutdown(wait=True)

    def test_start_noop_for_text_only_model(self):
        renderer = _make_renderer_mock({})
        renderer.mm_processor = None
        # _readonly_mm_processor is already None

        renderer.start_mm_warmup_in_background()

        assert renderer._mm_warmup_future is None

    def test_start_is_run_at_most_once(self):
        renderer = self._make_bg_renderer({"image": 1})
        try:
            with patch("vllm.multimodal.processing.TimingContext", autospec=True):
                renderer.start_mm_warmup_in_background()
                first = renderer._mm_warmup_future
                renderer.start_mm_warmup_in_background()  # must not spawn a second
                assert renderer._mm_warmup_future is first
                renderer._join_mm_warmup()
        finally:
            renderer._mm_executor.shutdown(wait=True)

    def test_warmup_joins_background_and_does_not_rerun_mm(self):
        # When a background MM warmup is in flight, warmup() must join it and
        # run only the chat warmup — the MM warmup must not run twice.
        renderer = self._make_bg_renderer({"image": 1})
        try:
            with patch("vllm.multimodal.processing.TimingContext", autospec=True):
                renderer.start_mm_warmup_in_background()
                BaseRenderer.warmup(renderer, ChatParams())

            # MM apply called exactly once (by the background warmup task).
            renderer.mm_processor.apply.assert_called_once()
            # Chat warmup ran exactly once.
            renderer.render_chat.assert_called_once()
            # Background task has been joined and cleared.
            assert renderer._mm_warmup_future is None
        finally:
            renderer._mm_executor.shutdown(wait=True)

    def test_warmup_does_not_rerun_mm_after_reset_joins_background(self):
        # Regression: reset_mm_cache joins the background warmup before
        # warmup() runs (it clears _mm_warmup_future). warmup() must still not
        # re-run the MM warmup — the _mm_warmup_done flag survives the join.
        renderer = self._make_bg_renderer({"image": 1})
        try:
            with patch("vllm.multimodal.processing.TimingContext", autospec=True):
                renderer.start_mm_warmup_in_background()
                # Simulate reset_mm_cache joining the background warmup first
                # (clears _mm_warmup_future, sets _mm_warmup_done via the join).
                renderer._join_mm_warmup()
                assert renderer._mm_warmup_future is None  # joined & cleared
                BaseRenderer.warmup(renderer, ChatParams())

            # MM apply called exactly once (by the background warmup task);
            # warmup() did not re-run warmup_mm despite the future being cleared.
            renderer.mm_processor.apply.assert_called_once()
            renderer.render_chat.assert_called_once()
        finally:
            renderer._mm_executor.shutdown(wait=True)

    def test_warmup_mm_runs_at_most_once(self):
        # Direct repeated calls to warmup_mm run the MM warmup only once.
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            renderer.warmup_mm()
            renderer.warmup_mm()  # second call must be a no-op

        renderer.mm_processor.apply.assert_called_once()

    def test_shutdown_joins_background_warmup(self):
        # shutdown() must join the background warmup before closing caches,
        # so the mm_processor_cache is never touched concurrently.
        renderer = self._make_bg_renderer({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            renderer.start_mm_warmup_in_background()
            future = renderer._mm_warmup_future
            renderer.shutdown()

        assert renderer._mm_warmup_future is None
        # The background warmup was allowed to complete (apply ran) and the
        # future is done after shutdown joined it.
        assert future.done()
        renderer.mm_processor.apply.assert_called_once()


class TestEngineStartWarmupHook:
    # The renderer is handed to the client, which starts the MM warmup
    # (renderer.start_mm_warmup_in_background) only after engine-core
    # processes have been forked — a live warmup thread at fork() time would
    # deadlock the child (it inherits locks whose owning thread vanishes).
    # These tests pin the renderer plumbing contract on EngineCoreClient.

    def _mock_config(self):
        from types import SimpleNamespace

        from vllm.v1.engine.core_client import EngineCoreClient  # noqa: F401

        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_external_lb=False,
            ),
            model_config=SimpleNamespace(multimodal_config=None),
        )
        return cfg

    def _mock_renderer(self):
        return MagicMock()

    def test_inproc_client_rejects_renderer(self):
        # InprocClient has no _start_mm_warmup (only MPClient does), so it
        # takes no renderer at all: passing one is a TypeError at the call
        # site rather than a silently-ignored kwarg.
        from vllm.v1.engine import core_client as cc

        with pytest.raises(TypeError, match="renderer"):
            cc.InprocClient(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                renderer=self._mock_renderer(),
            )

    def test_inproc_client_forwards_fail_callback_but_not_renderer(self):
        # EngineCore args (incl. executor_fail_callback) pass through; the
        # renderer kwarg is intercepted by InprocClient.__init__ (keyword-only
        # after *), so it is never forwarded to EngineCore — which has no
        # renderer parameter at all.
        from vllm.v1.engine import core_client as cc

        callback = MagicMock()
        with patch.object(cc, "EngineCore") as mock_engine_core:
            cc.InprocClient(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                executor_fail_callback=callback,
            )

        mock_engine_core.assert_called_once()
        args, kwargs = mock_engine_core.call_args
        assert len(args) == 3  # vllm_config, executor_class, log_stats
        assert "renderer" not in kwargs
        assert kwargs.get("executor_fail_callback") is callback

    def test_make_client_does_not_pass_renderer_to_inproc_client(self):
        # In-process: the renderer must never reach InprocClient, so MM
        # warmup stays a plain inline call inside renderer.warmup().
        from vllm.v1.engine import core_client as cc

        cfg = self._mock_config()
        renderer = self._mock_renderer()

        with (
            patch.object(cc, "InprocClient") as mock_inproc,
            patch.object(cc, "SyncMPClient") as mock_sync,
        ):
            client = cc.EngineCoreClient.make_client(
                multiprocess_mode=False,
                asyncio_mode=False,
                vllm_config=cfg,
                executor_class=MagicMock(),
                log_stats=False,
                renderer=renderer,
            )

        assert client is mock_inproc.return_value
        mock_inproc.assert_called_once()
        assert "renderer" not in mock_inproc.call_args.kwargs
        mock_sync.assert_not_called()
        renderer.start_mm_warmup_in_background.assert_not_called()

    @pytest.mark.parametrize(
        ("client_cls", "multiprocess_mode"),
        [
            # SyncMPClient forks engine-core: the renderer must be handed to
            # it so the warmup fires after proc.start(), not before fork().
            ("SyncMPClient", True),
        ],
    )
    def test_make_client_passes_renderer(self, client_cls, multiprocess_mode):
        from vllm.v1.engine import core_client as cc

        cfg = self._mock_config()
        renderer = self._mock_renderer()

        with patch.object(cc, client_cls) as mock_client:
            cc.EngineCoreClient.make_client(
                multiprocess_mode=multiprocess_mode,
                asyncio_mode=False,
                vllm_config=cfg,
                executor_class=MagicMock(),
                log_stats=False,
                renderer=renderer,
            )

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["renderer"] is renderer

    @pytest.mark.parametrize(
        ("client_cls", "dp_size", "external_lb"),
        [
            ("AsyncMPClient", 1, False),
            ("DPAsyncMPClient", 2, True),
        ],
    )
    def test_make_async_mp_client_passes_renderer(
        self, client_cls, dp_size, external_lb
    ):
        from vllm.v1.engine import core_client as cc

        cfg = self._mock_config()
        cfg.parallel_config.data_parallel_size = dp_size
        cfg.parallel_config.data_parallel_external_lb = external_lb
        renderer = self._mock_renderer()

        with patch.object(cc, client_cls) as mock_client:
            cc.EngineCoreClient.make_async_mp_client(
                cfg,
                executor_class=MagicMock(),
                log_stats=False,
                renderer=renderer,
            )

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["renderer"] is renderer

    def test_mp_client_starts_mm_warmup_only_with_renderer(self):
        # MPClient must call renderer.start_mm_warmup_in_background when a
        # renderer was given, and skip it (no crash) when not.
        from vllm.v1.engine import core_client as cc

        # No renderer: _start_mm_warmup is a no-op.
        client = cc.MPClient.__new__(cc.MPClient)
        client._renderer = None
        client._start_mm_warmup()

        # With a renderer: it starts the background MM warmup.
        renderer = self._mock_renderer()
        client2 = cc.MPClient.__new__(cc.MPClient)
        client2._renderer = renderer
        client2._start_mm_warmup()
        renderer.start_mm_warmup_in_background.assert_called_once()
