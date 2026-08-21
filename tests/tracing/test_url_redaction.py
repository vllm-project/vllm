import pytest
from typing import Any
from vllm.tracing.otel import is_otel_available

# Skip if OTel is missing
pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")

from vllm.tracing.otel import URLRedactingSpanProcessor

class MockSpan:
    def __init__(self, attributes):
        self.attributes = attributes
        self._attributes = attributes # simulate internal dict

def test_url_redaction():
    processor = URLRedactingSpanProcessor()
    
    # Test http.url
    attributes = {"http.url": "https://example.com/image.jpg?token=secret"}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["http.url"] == "https://example.com/image.jpg"
    
    # Test url.full
    attributes = {"url.full": "https://example.com/image.jpg?token=secret&other=bar"}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["url.full"] == "https://example.com/image.jpg"
    
    # Test http.target
    attributes = {"http.target": "/v1/completions?token=secret"}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["http.target"] == "/v1/completions"
    
    # Test url.query
    attributes = {"url.query": "token=secret"}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["url.query"] == ""
    
    # Test no query params
    attributes = {"http.url": "https://example.com/image.jpg"}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["http.url"] == "https://example.com/image.jpg"
    
    # Test non-string
    attributes = {"http.url": 123}
    span = MockSpan(attributes)
    processor.on_end(span)
    assert span.attributes["http.url"] == 123
