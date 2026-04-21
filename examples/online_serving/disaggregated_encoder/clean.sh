kill -9 $(ps aux | grep -iE VLLM | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep server.py | awk '{ print $2 }') 2>/dev/null
kill -9 $(ps aux | grep python | awk '{ print $2 }') 2>/dev/null
kill -9 $(ps aux | grep python3.9 | awk '{ print $2 }') 2>/dev/null


kill -9 $(ps aux | grep "vllm serve" | awk '{ print $2 }') 2>/dev/null
kill -9 $(ps aux | grep disagg_proxy_demo.py | awk '{ print $2 }') 2>/dev/null


kill -9 $(ps aux | grep "bash /workspace/hero/" | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep "[VLLM::APIServer]" | awk '{ print $2 }') 2>/dev/null


kill -9 $(ps aux | grep timeout | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep until | awk '{ print $2 }') 2>/dev/null
kill -9 $(ps aux | grep VLLM::EngineCoreProc_0 | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep -iE VLLM | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep -iE redis | awk '{ print $2 }') 2>/dev/null

kill -9 $(ps aux | grep -iE timeout | awk '{ print $2 }') 2>/dev/null
kill -9 $(ps aux | grep -iE sleep | awk '{ print $2 }') 2>/dev/null

# Cleanup commands
pgrep python | xargs kill -9
pkill -f python