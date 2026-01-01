import itertools
from vllm.logprobs import FlatLogprobs, Logprob


def test_flat_logprobs_append_and_get():
    f = FlatLogprobs()
    # first position (prompt) - None per create_prompt_logprobs semantics
    f.append(None)

    # second position: two token candidates
    token_ids = [10, 20]
    logprobs = [ -0.1, -2.3 ]
    ranks = itertools.chain((1,), (1, 2))
    decoded = ["a", "b"]
    f.append_fast(token_ids, logprobs, ranks, decoded)

    # verify length and contents
    assert len(f) == 2
    pos1 = f[0]
    assert pos1 == {}
    pos2 = f[1]
    assert 10 in pos2 and 20 in pos2
    assert pos2[10].logprob == -0.1

    # negative index should work
    pos2_neg = f[-1]
    assert pos2_neg[20].decoded_token == "b"


def test_flat_logprobs_slice():
    f = FlatLogprobs()
    f.append(None)
    f.append_fast([1], [0.0], itertools.chain((1,), (1,)), ["x"])
    f.append_fast([2], [0.0], itertools.chain((1,), (1,)), ["y"])

    # slice should return a FlatLogprobs containing the slice
    s = f[1:3]
    assert isinstance(s, FlatLogprobs)
    assert len(s) == 2
    assert s.token_ids == [1, 2]
