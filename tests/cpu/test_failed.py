def test_failed():
    print("This test should fail in CI.")
    raise RuntimeError("This test should fail in CI.")
