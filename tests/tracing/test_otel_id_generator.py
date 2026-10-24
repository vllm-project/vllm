# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm.tracing.otel import is_otel_available

pytestmark = pytest.mark.skipif(
    not is_otel_available(), reason="OpenTelemetry is not installed"
)


def test_secrets_id_generator_ignores_random_seed():
    from vllm.tracing.otel import SecretsIdGenerator

    gen = SecretsIdGenerator()

    random.seed(0)
    span_id_1 = gen.generate_span_id()
    trace_id_1 = gen.generate_trace_id()

    random.seed(0)
    span_id_2 = gen.generate_span_id()
    trace_id_2 = gen.generate_trace_id()

    assert span_id_1 != span_id_2
    assert trace_id_1 != trace_id_2
    assert span_id_1 != 0
    assert trace_id_1 != 0
