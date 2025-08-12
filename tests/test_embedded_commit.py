import vllm


def test_embedded_commit_defined():
    assert hasattr(vllm, "__version__")
    assert hasattr(vllm, "__version_tuple__")
    assert vllm.__version__ != "dev"
    assert vllm.__version_tuple__ != (0, 0, "dev")
