"""
training/reward.py

Milestone-based reward function from the CUDA-Agent paper.

The paper uses discrete milestone rewards rather than raw speedup ratios
to avoid outlier sensitivity and reward-hacking. Ablation studies showed
that discrete rewards significantly outperform continuous rewards:
  - Discrete: 96.8% faster-than-compile rate, 2.11x speedup
  - Continuous: 60.4% faster-than-compile rate, 1.25x speedup

Reward milestones:
  -1 : Compilation or correctness failure
  +1 : Correct output, but no speedup milestone reached
  +2 : Correct + faster than eager execution (but not torch.compile)
  +3 : Correct + ≥5% speedup over torch.compile (primary target)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda_agent.agent.environment import EpisodeState, Reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward constants (match paper Table 1)
# ---------------------------------------------------------------------------

REWARD_FAILURE         = -1.0
REWARD_CORRECT         = 1.0
REWARD_FASTER_EAGER    = 2.0
REWARD_FASTER_COMPILE  = 3.0   # target: ≥5% speedup over torch.compile

COMPILE_SPEEDUP_THRESHOLD = 0.95   # extension_time ≤ 0.95 × compile_time


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    value: float
    milestone: str
    speedup_vs_eager: float = 0.0
    speedup_vs_compile: float = 0.0
    verified: bool = False


def compute_milestone_reward(
    verified: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> RewardResult:
    """
    Compute the discrete milestone reward from profiling results.

    Args:
        verified:           Did the model pass correctness verification?
        speedup_vs_eager:   (eager_time / extension_time) ratio
        speedup_vs_compile: (compile_time / extension_time) ratio

    Returns:
        RewardResult with the milestone reward value and description.
    """
    if not verified:
        return RewardResult(
            value=REWARD_FAILURE,
            milestone="correctness_failure",
            verified=False,
        )

    if speedup_vs_compile >= 1.0 / COMPILE_SPEEDUP_THRESHOLD:
        # extension_time ≤ 95% of compile_time  →  ≥5.26% speedup
        return RewardResult(
            value=REWARD_FASTER_COMPILE,
            milestone="faster_than_compile",
            speedup_vs_eager=speedup_vs_eager,
            speedup_vs_compile=speedup_vs_compile,
            verified=True,
        )

    if speedup_vs_eager >= 1.0:
        return RewardResult(
            value=REWARD_FASTER_EAGER,
            milestone="faster_than_eager",
            speedup_vs_eager=speedup_vs_eager,
            speedup_vs_compile=speedup_vs_compile,
            verified=True,
        )

    return RewardResult(
        value=REWARD_CORRECT,
        milestone="correct_no_speedup",
        speedup_vs_eager=speedup_vs_eager,
        speedup_vs_compile=speedup_vs_compile,
        verified=True,
    )


def reward_from_episode_state(state: "EpisodeState") -> RewardResult:
    """Convenience wrapper: compute reward from a completed EpisodeState."""
    from cuda_agent.agent.environment import Reward as R
    milestone_map = {
        R.CORRECTNESS_FAILURE: "correctness_failure",
        R.CORRECT_NO_SPEEDUP:  "correct_no_speedup",
        R.FASTER_THAN_EAGER:   "faster_than_eager",
        R.FASTER_THAN_COMPILE: "faster_than_compile",
    }
    value_map = {
        R.CORRECTNESS_FAILURE: REWARD_FAILURE,
        R.CORRECT_NO_SPEEDUP:  REWARD_CORRECT,
        R.FASTER_THAN_EAGER:   REWARD_FASTER_EAGER,
        R.FASTER_THAN_COMPILE: REWARD_FASTER_COMPILE,
    }
    return RewardResult(
        value=value_map[state.reward],
        milestone=milestone_map[state.reward],
        speedup_vs_compile=state.speedup_vs_compile,
        verified=state.verified,
    )


# ---------------------------------------------------------------------------
# Advantage computation (GAE — Generalised Advantage Estimation)
# Paper: γ=1, λ=0.95 for critic pretraining
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 1.0,
    lam: float = 0.95,
) -> list[float]:
    """
    Compute Generalised Advantage Estimation (GAE) for a trajectory.

    Paper: γ=1 (no discounting — sparse terminal reward), λ=0.95

    Args:
        rewards: Per-step reward list (typically 0.0 except at terminal step).
        values:  Critic value estimates for each step.
        gamma:   Discount factor (paper uses γ=1).
        lam:     GAE λ parameter (paper uses λ=0.95).

    Returns:
        List of advantage estimates, one per step.
    """
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]

    return advantages


def compute_returns(
    rewards: list[float],
    gamma: float = 1.0,
) -> list[float]:
    """Monte-Carlo returns (for critic pre-training target values)."""
    returns = [0.0] * len(rewards)
    ret = 0.0
    for t in reversed(range(len(rewards))):
        ret = rewards[t] + gamma * ret
        returns[t] = ret
    return returns
