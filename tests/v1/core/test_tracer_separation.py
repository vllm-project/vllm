"""Test that journey and step tracers are properly separated with distinct scopes."""
import pytest
from unittest.mock import Mock, patch


class TestTracerSeparation:
    """Verify journey and step tracers use distinct scopes."""

    def test_separate_tracers_distinct_scopes(self):
        """Verify journey and step tracers are initialized with different scope names.

        Journey tracing should use "vllm.scheduler" scope (for llm_core spans).
        Step tracing should use "vllm.scheduler.step" scope (for scheduler_steps span).

        This ensures they appear as distinct services in Jaeger.
        """
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Verify journey tracer uses vllm.scheduler scope
        assert 'init_tracer("vllm.scheduler", endpoint)' in content
        assert 'self.journey_tracer = init_tracer("vllm.scheduler"' in content

        # Verify step tracer uses vllm.scheduler.step scope
        assert 'init_tracer("vllm.scheduler.step", endpoint)' in content
        assert 'self.step_tracer = init_tracer("vllm.scheduler.step"' in content

    def test_no_shared_tracer_variable(self):
        """Verify that self.tracer is no longer used (replaced with specific tracers)."""
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Count occurrences of self.tracer (excluding self.journey_tracer and self.step_tracer)
        lines = content.split('\n')
        plain_tracer_lines = [
            line for line in lines
            if 'self.tracer' in line
            and 'self.journey_tracer' not in line
            and 'self.step_tracer' not in line
        ]

        # There should be NO references to plain self.tracer
        assert len(plain_tracer_lines) == 0, f"Found unexpected self.tracer references: {plain_tracer_lines}"

    def test_both_tracers_declared(self):
        """Verify both journey_tracer and step_tracer are declared as instance variables."""
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Both should be declared
        assert "self.journey_tracer: Any | None = None" in content
        assert "self.step_tracer: Any | None = None" in content

    def test_journey_tracer_used_for_core_spans(self):
        """Verify that llm_core spans use journey_tracer, not step_tracer."""
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Check _create_core_span method
        assert 'if not self.journey_tracer:' in content
        assert 'self.journey_tracer.start_span(' in content

    def test_step_tracer_used_for_step_spans(self):
        """Verify that scheduler_steps span uses step_tracer, not journey_tracer."""
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Check step span creation
        assert 'if self._enable_step_tracing and self.step_tracer is not None:' in content
        assert 'self._step_span = self.step_tracer.start_span(' in content

    def test_no_accidental_coupling(self):
        """Verify step tracing doesn't check if journey_tracer exists.

        Old design had: if self.tracer is None (coupling step to journey)
        New design: each feature initializes its own tracer independently
        """
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Step tracing init should not check journey_tracer
        step_init_section = content[content.find("if self._enable_step_tracing:"):content.find("# Create long-lived scheduler_steps span")]

        # Should NOT contain references to journey_tracer in step init
        assert "self.journey_tracer" not in step_init_section

    def test_both_share_same_provider_singleton(self):
        """Verify both tracers use init_tracer, which shares singleton provider.

        The singleton provider in init_tracer() ensures both tracers export to
        the same OTLP endpoint, even though they use different scope names.
        """
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Both should call init_tracer (which uses the singleton)
        journey_init = content.find('self.journey_tracer = init_tracer("vllm.scheduler"')
        step_init = content.find('self.step_tracer = init_tracer("vllm.scheduler.step"')

        # Both should exist
        assert journey_init != -1
        assert step_init != -1

        # Both should use the same endpoint variable
        assert "endpoint" in content[journey_init:journey_init+100]
        assert "endpoint" in content[step_init:step_init+100]

    def test_comments_explain_separation(self):
        """Verify comments explain the distinct scope design."""
        # Read the scheduler code
        with open("vllm/v1/core/sched/scheduler.py") as f:
            content = f.read()

        # Should have comment explaining separation
        assert "Each feature gets its own tracer with distinct scope" in content
        assert "clean separation in Jaeger" in content
        assert "vllm.scheduler.step" in content  # Mention the step scope
