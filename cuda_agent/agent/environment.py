"""
agent/environment.py

Sandboxed execution environment for the CUDA-Agent.

Provides the compile → verify → profile loop described in the paper,
with anti-reward-hacking safeguards:
  - File permission controls (protect utils/, binding.cpp, binding_registry.h)
  - Isolated working directories per episode
  - Device synchronisation and warm-up for reliable profiling
  - 5-pass correctness validation

Paper reward milestones:
  -1 : Correctness failure
  +1 : Correct (no speedup milestone reached)
  +2 : Faster than eager execution
  +3 : Faster than both eager AND torch.compile (≥5% speedup)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward enum (milestone-based, as described in paper)
# ---------------------------------------------------------------------------

class Reward(IntEnum):
    CORRECTNESS_FAILURE = -1
    CORRECT_NO_SPEEDUP  = 1
    FASTER_THAN_EAGER   = 2
    FASTER_THAN_COMPILE = 3   # ≥5% speedup over torch.compile — target reward


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    elapsed_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_observation(self) -> str:
        parts = []
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout[:4096]}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr[:2048]}")
        if not self.success:
            parts.append(f"[Exit code: {self.returncode}]")
        if self.elapsed_s > 0:
            parts.append(f"[Elapsed: {self.elapsed_s:.2f}s]")
        return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROTECTED_PATHS = {
    "utils",
    "binding.cpp",
    "binding_registry.h",
}

COMPILE_CMD = "TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh"
VERIFY_CMD  = "python3 -m utils.verification"
PROFILE_CMD = "python3 -m utils.profiling --iters 20"


@dataclass
class EpisodeState:
    """Tracks the state of a single agent episode."""
    task_dir: Path
    turn: int = 0
    compiled: bool = False
    verified: bool = False
    reward: Reward = Reward.CORRECTNESS_FAILURE
    speedup_vs_compile: float = 0.0
    done: bool = False
    history: list[dict[str, str]] = field(default_factory=list)


class CUDAAgentEnvironment:
    """
    Manages the agent_workdir sandbox for a single optimization episode.

    Usage:
        env = CUDAAgentEnvironment(template_dir=..., cuda_arch="9.0")
        state = env.reset(sample)
        while not state.done:
            action, action_input = agent.step(state)
            obs, state = env.step(state, action, action_input)
        reward = state.reward
    """

    MAX_TURNS = 150          # paper: up to 150 training turns, 200 at eval
    MAX_FILE_SIZE = 1 << 20  # 1 MB — limit individual kernel file size
    TIMEOUT_S = 300          # 5 min per tool call

    def __init__(
        self,
        template_dir: str | Path,
        cuda_arch: str = "9.0",
        work_root: str | Path | None = None,
        max_turns: int = MAX_TURNS,
    ):
        self.template_dir = Path(template_dir)
        self.cuda_arch = cuda_arch
        self.work_root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="cuda_agent_"))
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_dir: str | Path) -> EpisodeState:
        """
        Set up an isolated working directory for a new episode.

        Args:
            task_dir: Path to the pre-generated task directory
                      (created by TaskGenerator.generate_task).

        Returns:
            Initial EpisodeState.
        """
        ep_dir = self.work_root / f"ep_{int(time.time()*1000)}"
        shutil.copytree(str(task_dir), str(ep_dir))
        self._apply_file_protections(ep_dir)

        state = EpisodeState(task_dir=ep_dir)
        logger.info("Episode started: %s", ep_dir)
        return state

    def step(
        self,
        state: EpisodeState,
        action: str,
        action_input: dict | str,
    ) -> tuple[str, EpisodeState]:
        """
        Execute one agent action and return (observation, updated_state).

        Supported actions: bash, write_file, read_file.
        """
        if state.done:
            return "Episode is already done.", state

        state.turn += 1
        if state.turn > self.max_turns:
            state.done = True
            obs = f"[TIMEOUT] Maximum turns ({self.max_turns}) reached."
            return obs, state

        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except json.JSONDecodeError:
                action_input = {"command": action_input}

        # Dispatch
        if action == "bash":
            result = self._run_bash(state, action_input.get("command", ""))
        elif action == "write_file":
            result = self._write_file(state,
                                      action_input.get("path", ""),
                                      action_input.get("content", ""))
        elif action == "read_file":
            result = self._read_file(state, action_input.get("path", ""))
        else:
            result = ToolResult(stderr=f"Unknown action: {action!r}", returncode=1)

        obs = result.to_observation()

        # After every bash call, check if we should auto-detect completion
        if action == "bash":
            self._update_episode_state(state, action_input.get("command", ""), obs)

        # Record in history
        state.history.append({"role": "assistant", "content": f"Action: {action}\n{json.dumps(action_input)}"})
        state.history.append({"role": "user", "content": f"Observation: {obs}"})

        return obs, state

    def cleanup(self, state: EpisodeState) -> None:
        """Remove the temporary episode directory."""
        if state.task_dir.exists():
            shutil.rmtree(state.task_dir)
            logger.info("Cleaned up episode dir: %s", state.task_dir)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _run_bash(self, state: EpisodeState, command: str) -> ToolResult:
        """Execute a bash command in the task directory."""
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = self.cuda_arch

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(state.task_dir),
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_S,
                env=env,
            )
            elapsed = time.perf_counter() - t0
            return ToolResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                elapsed_s=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                stderr=f"Command timed out after {self.TIMEOUT_S}s.",
                returncode=1,
                elapsed_s=self.TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(stderr=str(exc), returncode=1)

    def _write_file(
        self, state: EpisodeState, path: str, content: str
    ) -> ToolResult:
        """Write content to a file, enforcing protected-path and size limits."""
        # Security: block writes to protected infrastructure
        rel_path = Path(path)
        for protected in PROTECTED_PATHS:
            if str(rel_path).startswith(protected):
                return ToolResult(
                    stderr=f"[BLOCKED] Writing to {path!r} is not allowed.",
                    returncode=1,
                )

        if len(content) > self.MAX_FILE_SIZE:
            return ToolResult(
                stderr=f"File too large: {len(content)} bytes (max {self.MAX_FILE_SIZE}).",
                returncode=1,
            )

        target = state.task_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(stdout=f"Written {len(content)} bytes to {path}")

    def _read_file(self, state: EpisodeState, path: str) -> ToolResult:
        """Read a file from the task directory."""
        target = state.task_dir / Path(path)
        if not target.exists():
            return ToolResult(stderr=f"File not found: {path}", returncode=1)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(stdout=content[:8192])  # truncate very large files
        except Exception as exc:  # noqa: BLE001
            return ToolResult(stderr=str(exc), returncode=1)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(self, state: EpisodeState) -> Reward:
        """
        Evaluate the final state and return a milestone reward.

        Paper reward structure:
          -1 : Correctness failure
          +1 : Correct, no speedup milestone
          +2 : Correct + faster than eager
          +3 : Correct + faster than torch.compile (≥5%)
        """
        if not state.compiled:
            return Reward.CORRECTNESS_FAILURE

        verify_result = self._run_bash(state, VERIFY_CMD)
        if not verify_result.success or "FAILED" in verify_result.stdout:
            logger.info("Verification failed for episode %s", state.task_dir)
            return Reward.CORRECTNESS_FAILURE

        # Parse profiling output
        profile_result = self._run_bash(state, PROFILE_CMD)
        speedup = _parse_speedup_vs_compile(profile_result.stdout)

        if speedup is None:
            return Reward.CORRECT_NO_SPEEDUP

        state.speedup_vs_compile = speedup
        if speedup >= 1.05:   # ≥5% speedup over torch.compile
            return Reward.FASTER_THAN_COMPILE
        if speedup >= 1.0:    # faster than eager (but not compiled)
            return Reward.FASTER_THAN_EAGER
        return Reward.CORRECT_NO_SPEEDUP

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_file_protections(self, task_dir: Path) -> None:
        """Make protected files read-only to prevent agent tampering."""
        for protected in PROTECTED_PATHS:
            target = task_dir / protected
            if target.is_file():
                target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            elif target.is_dir():
                for f in target.rglob("*"):
                    if f.is_file():
                        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    def _update_episode_state(
        self, state: EpisodeState, command: str, obs: str
    ) -> None:
        """Update state flags based on command output."""
        if "compile.sh" in command or "utils.compile" in command:
            if "cuda_extension.so written" in obs or "Exit code: 0" not in obs[:100]:
                state.compiled = "[ERROR]" not in obs
        if "utils.verification" in command or "verification" in command:
            if "All" in obs and "succeeded" in obs:
                state.verified = True
                state.done = False  # keep going to optimize further
            elif "FAILED" in obs:
                state.verified = False


def _parse_speedup_vs_compile(profiling_output: str) -> float | None:
    """
    Parse speedup vs. torch.compile from profiling output.

    Expected format:
      Torch Baseline: Xus, Torch Compile: Yus, CUDA Extension: Zus
    """
    import re
    pattern = r"Torch Compile:\s*([\d.]+)us.*CUDA Extension:\s*([\d.]+)us"
    m = re.search(pattern, profiling_output)
    if m:
        compile_us = float(m.group(1))
        extension_us = float(m.group(2))
        if extension_us > 0:
            return compile_us / extension_us
    # Also parse explicit speedup line
    speedup_pattern = r"Speedup vs torch\.compile:\s*([\d.]+)x"
    m2 = re.search(speedup_pattern, profiling_output)
    if m2:
        return float(m2.group(1))
    return None
