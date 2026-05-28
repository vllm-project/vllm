tb run \
    --agent-import-path terminal_agents.CmdQwenTerminalAgent:CmdQwenTerminalAgent \
    --model Qwen/Qwen3-4B \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --n-concurrent 1 \
    --log-level debug

# For specific task, add --task-id argument, e.g.,
#     --task-id hello-world
