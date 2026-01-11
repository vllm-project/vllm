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


def test_flat_logprobs_empty_positions_and_step_slice():
    f = FlatLogprobs()
    # position 0: empty (None)
    f.append(None)
    # position 1: has tokens
    f.append_fast([5, 6], [-0.5, -1.2], itertools.chain((1,), (1, 2)), ["a", "b"])
    # position 2: empty again
    f.append(None)
    # position 3: has tokens
    f.append_fast([7], [-0.3], itertools.chain((1,), (1,)), ["c"])

    # verify empty positions yield empty dict
    assert f[0] == {}
    assert f[2] == {}

    # step slice: select positions 1 and 3 using step=2
    s = f[1:4:2]
    assert isinstance(s, FlatLogprobs)
    # expect token_ids concatenated from positions 1 and 3
    assert s.token_ids == [5, 6, 7]


def test_flat_logprobs_out_of_range_index():
    f = FlatLogprobs()
    f.append(None)
    f.append_fast([1], [0.0], itertools.chain((1,), (1,)), ["x"])
    # positive out of range
    try:
        _ = f[10]
        raise AssertionError("Expected IndexError for out-of-range positive index")
    except IndexError:
        pass
    # negative out of range
    try:
        _ = f[-10]
        raise AssertionError("Expected IndexError for out-of-range negative index")
    except IndexError:
        pass


def test_flat_logprobs_large_data_performance():
    f = FlatLogprobs()
    # create many positions with a small number of tokens each
    N = 10000
    f.append(None)
    for i in range(1, N):
        f.append_fast([i], [float(-i)], itertools.chain((1,), (1,)), [str(i)])
    # sanity checks
    assert len(f) == N
    # access last position
    last = f[-1]
    # token id should be N-1
    assert (N - 1) in last
