from pathlib import Path
import json

from openai import OpenAI
from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run one shell command in /app. Use for all file/shell work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Single shell command, e.g. printf 'Hello, world!\\n' > hello.txt",
                    },
                    "block": {
                        "type": "boolean",
                        "description": "Wait for command to finish before next step",
                        "default": True,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_task_complete",
            "description": "Call when tests should pass; no more commands needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

class CmdQwenTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "cmd-qwen-terminal-agent"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        api_base: str = "http://localhost:7557/v1",
        api_key: str = "EMPTY",
        max_turns: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model_name = model_name
        self._max_turns = max_turns
    def _dispatch_tool(self, name: str, args: dict, session: TmuxSession) -> str:
        if name == "run_shell_command":
            cmd = args["command"]
            session.send_keys([cmd, "Enter"], block=args.get("block", True))
            return session.capture_pane()[-4000:]  # truncate if needed
        if name == "mark_task_complete":
            return "ok"
        return f"unknown tool: {name}"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        model = self._client.models.list().data[0].id
        messages = [
            {
                "role": "system",
                "content": (
                    "You solve terminal tasks by calling tools. "
                    "Work in /app. One command per run_shell_command call."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{instruction}\n\n"
                    f"Terminal:\n{session.capture_pane()}"
                ),
            },
        ]
        for _ in range(self._max_turns):
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            msg = resp.choices[0].message
            # No tools → model answered in plain text (still a vLLM call, not tool path)
            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
            # Assistant turn with tool_calls (this is what you're benchmarking)
            messages.append({
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
            })
            done = False
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._dispatch_tool(tc.function.name, args, session)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if tc.function.name == "mark_task_complete":
                    done = True
            if done:
                break
        return AgentResult(failure_mode=FailureMode.NONE)
