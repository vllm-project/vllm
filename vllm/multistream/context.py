from contextlib import contextmanager
from typing import Any

_ms_comm_context: Any = None
_ms_layer_index_context: int = -1
_ms_metadata_context: Any = None

def set_multistream_layer_context(start_layer: int, ms_metadata: Any):
    """
    set multistream layer context before transformer layers
    """
    global _ms_layer_index_context, _ms_metadata_context
    _ms_layer_index_context = start_layer
    _ms_metadata_context = ms_metadata

def reset_multistream_layer_context():
    """
    reset multistream layer context
    """
    global _ms_layer_index_context, _ms_metadata_context
    _ms_layer_index_context = -1
    _ms_metadata_context = None

def get_multistream_layer_context():
    """
    get multistream layer context
    """
    return _ms_layer_index_context, _ms_metadata_context

def advance_step_multistream_layer_context():
    """
    advance multistream layer index context
    """
    global _ms_layer_index_context
    _ms_layer_index_context += 1


def get_multistream_comm_context() -> Any:
    """Get the current comm forward context."""
    return _ms_comm_context

@contextmanager
def set_multistream_context(context: Any):
    """A context manager that stores the current comm forward context,
    can be attention metadata, etc."""
    global _ms_comm_context
    _ms_comm_context = context
    try:
        yield
    finally:
        _ms_comm_context = None

# for TCCL

_in_cuda_graph_capture: bool = False

def set_cuda_graph_capture_state():
    global _in_cuda_graph_capture
    _in_cuda_graph_capture = True

def get_cuda_graph_capture_state():
    global _in_cuda_graph_capture
    return _in_cuda_graph_capture

def reset_cuda_graph_capture_state():
    global _in_cuda_graph_capture
    _in_cuda_graph_capture = False
