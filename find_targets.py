import ast
import os
import glob

targets = []

for filepath in glob.glob("vllm/model_executor/models/*.py"):
    with open(filepath, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            continue
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.endswith("ForCausalLM"):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "load_weights":
                        source = ast.unparse(item)
                        if "AutoWeightsLoader" not in source:
                            targets.append((filepath, node.name))

print("Found", len(targets), "targets")
for f, c in targets:
    print(f, c)

