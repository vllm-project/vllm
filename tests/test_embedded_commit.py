import vllm


def test_embedded_commit_defined():
    assert len(vllm.__commit__) > 7
