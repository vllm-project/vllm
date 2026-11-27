from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run one shell command in /app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_task_complete",
            "description": "Call when the task is done.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class CmdQwenHarborAgent(BaseAgent):
    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str = "http://localhost:7557/v1",
        api_key: str = "EMPTY",
        max_turns: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "cmd-qwen-harbor"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]

        for _ in range(self._max_turns):
            resp = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if tc.function.name == "run_shell_command":
                    result = await environment.exec(args["command"], cwd="/app")
                    out = result.stdout or result.stderr or ""
                    if result.return_code != 0:
                        out = f"{out}\n[exit code: {result.return_code}]".strip()
                else:
                    out = "ok"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": out,
                    }
                )

                if tc.function.name == "mark_task_complete":
                    return

        if resp.usage:
            context.n_input_tokens = resp.usage.prompt_tokens
            context.n_output_tokens = resp.usage.completion_tokens
