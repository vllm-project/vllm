"""CPU checks for the cost fit and the chunk policy. No GPU and no vLLM."""

from prefill_budget.cost_model import ChunkPolicy, fit_cost


def _true_ms(q: int, c: int) -> float:
    # Scaled-basis truth: a=4, b=6, g=8, d=3, SCALE=1000.
    u = q / 1000.0
    v = c / 1000.0
    return 4.0 * u * u + 6.0 * u * v + 8.0 * u + 3.0


def test_fit_recovers_known_quadratic() -> None:
    samples = [
        (q, c, _true_ms(q, c))
        for q in (128, 256, 512, 1024, 1536, 2048)
        for c in (0, 1024, 4096, 8192)
    ]
    fit = fit_cost(samples)
    assert fit is not None
    worst = max(abs(fit.predict_ms(q, c) - truth) for q, c, truth in samples)
    assert worst < 1e-4, worst
    assert abs(fit.a - 4.0) < 1e-6
    assert abs(fit.b - 6.0) < 1e-6


def test_longer_prefix_gets_a_shorter_chunk() -> None:
    samples = [
        (q, c, _true_ms(q, c))
        for q in (128, 256, 512, 1024, 1536, 2048)
        for c in (0, 1024, 4096, 8192)
    ]
    fit = fit_cost(samples)
    assert fit is not None
    budget = 30.0
    q_short = fit.solve_q(0, budget, 128, 2048)
    q_long = fit.solve_q(8192, budget, 128, 2048)
    assert q_long < q_short
    assert fit.predict_ms(q_long, 8192) <= budget
    assert fit.predict_ms(q_long + 1, 8192) > budget
    assert fit.solve_q(0, budget_ms=0.1, q_min=128, q_max=2048) == 128


def test_negative_quadratic_term_is_dropped() -> None:
    # Times fall as q^2 grows, which a non-negative model must not copy.
    samples = [(q, 0, 50.0 - (q / 1000.0) ** 2) for q in (128, 256, 512, 1024, 2048)]
    fit = fit_cost(samples)
    assert fit is not None
    assert fit.a == 0.0


def test_policy_warms_up_then_shrinks_the_chunk() -> None:
    policy = ChunkPolicy(
        mode="budget",
        budget_ms=30.0,
        fixed_chunk=1024,
        min_chunk=128,
        max_chunk=2048,
        warmup_steps=2,
        block_size=16,
    )
    for _ in range(2):
        q, phase = policy.cap(0, remaining=8000)
        assert phase == "warmup"
        assert policy.observe(q, 0, _true_ms(q, 0))
    for q in (128, 512, 1024, 2048):
        for c in (0, 4096, 8192):
            policy.observe(q, c, _true_ms(q, c))
    assert policy.fit is not None
    q_early, phase_early = policy.cap(0, remaining=8000)
    q_late, phase_late = policy.cap(8192, remaining=8000)
    assert phase_early == "budget"
    assert phase_late == "budget"
    assert q_late < q_early


def test_fixed_mode_ignores_the_clock() -> None:
    policy = ChunkPolicy(
        mode="fixed",
        budget_ms=30.0,
        fixed_chunk=1000,
        min_chunk=128,
        max_chunk=2048,
        warmup_steps=0,
        block_size=64,
    )
    q, phase = policy.cap(0, remaining=8000)
    assert phase == "fixed"
    assert q == 960  # 1000 rounded down to the 64-token block
    assert policy.observe(q, 0, 10.0) is True
    assert policy.fit is None


def test_slow_steps_do_not_enter_the_fit() -> None:
    policy = ChunkPolicy(
        mode="budget",
        budget_ms=30.0,
        fixed_chunk=1024,
        min_chunk=128,
        max_chunk=2048,
        warmup_steps=0,
        block_size=16,
    )
    for q in (128, 256, 512, 1024):
        assert policy.observe(q, 0, 5000.0) is False
    assert policy.fit is None


if __name__ == "__main__":
    test_fit_recovers_known_quadratic()
    test_longer_prefix_gets_a_shorter_chunk()
    test_negative_quadratic_term_is_dropped()
    test_policy_warms_up_then_shrinks_the_chunk()
    test_fixed_mode_ignores_the_clock()
    test_slow_steps_do_not_enter_the_fit()
    print("6 checks passed")
