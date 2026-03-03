"""CUDA-Agent RL training module."""
from cuda_agent.training.reward import (
    compute_milestone_reward,
    compute_gae,
    compute_returns,
    RewardResult,
    REWARD_FAILURE,
    REWARD_CORRECT,
    REWARD_FASTER_EAGER,
    REWARD_FASTER_COMPILE,
)
from cuda_agent.training.rft import (
    filter_trajectories,
    build_sft_dataset,
    run_rft,
)
from cuda_agent.training.rl_trainer import (
    PPOConfig,
    AgenticPPOTrainer,
    collect_rollouts,
    compute_ppo_loss,
    compute_value_loss,
    pretrain_critic,
)

__all__ = [
    "compute_milestone_reward",
    "compute_gae",
    "compute_returns",
    "RewardResult",
    "REWARD_FAILURE",
    "REWARD_CORRECT",
    "REWARD_FASTER_EAGER",
    "REWARD_FASTER_COMPILE",
    "filter_trajectories",
    "build_sft_dataset",
    "run_rft",
    "PPOConfig",
    "AgenticPPOTrainer",
    "collect_rollouts",
    "compute_ppo_loss",
    "compute_value_loss",
    "pretrain_critic",
]
