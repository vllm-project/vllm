import vllm


def test_embedded_commit_defined():
    assert vllm.__commit__ != "COMMIT_HASH_PLACEHOLDER"
    # 7 characters is the length of a short commit hash
    assert len(vllm.__commit__) >= 7
