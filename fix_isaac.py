import re

file_path = "/Users/jashwanth/Desktop/ai/vLLM/vllm/model_executor/models/isaac.py"
with open(file_path, "r") as f:
    content = f.read()

# Remove __future__ import
content = content.replace("from __future__ import annotations\n", "")

# Add typing imports
if "from typing import " in content:
    content = content.replace("from typing import ", "from typing import Optional, Union, ", 1)

# Replace X | None with Optional[X]
content = re.sub(r'\b([A-Za-z0-9_.]+(?:\[[^\]]*\])?)\s*\|\s*None\b', r'Optional[\1]', content)

# Replace torch.Tensor | IntermediateTensors
content = content.replace("torch.Tensor | IntermediateTensors", "Union[torch.Tensor, IntermediateTensors]")

with open(file_path, "w") as f:
    f.write(content)

print("File updated.")
