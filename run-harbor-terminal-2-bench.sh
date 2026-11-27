harbor run -d "terminal-bench/terminal-bench-2" \
  --agent-import-path terminal_agents.CmdQwenHarborAgent:CmdQwenHarborAgent \
  -m "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --ak api_base=http://localhost:7557/v1 \
  --debug

# For specific task, add -i argument, e.g.,
#     -i "terminal-bench/fix-git"
