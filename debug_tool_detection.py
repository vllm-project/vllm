from vllm.entrypoints.openai.tool_parsers.minimax_tool_parser import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# 创建parser
tokenizer = get_tokenizer('MiniMaxAi/MiniMax-M1-40k')
parser = MinimaxToolParser(tokenizer)

# 测试工具内容提取
test_cases = [
    '{"name": "tool1", "arguments": {"a": 1}}',
    '{"name": "tool1", "arguments": {"a": 1}}\n{"name": "tool2"',
    '{"name": "tool1", "arguments": {"a": 1}}\n{"name": "tool2", "arguments": {"b": 2}}'
]

for i, content in enumerate(test_cases):
    print(f"\n=== Test case {i+1} ===")
    print(f"Content: {repr(content)}")
    
    # 测试边界检测
    boundaries = parser._find_tool_boundaries(content)
    print(f"Boundaries: {boundaries}")
    
    # 测试工具数量
    tool_count = parser._detect_tools_in_text(content)
    print(f"Tool count: {tool_count}")
    
    # 测试每个工具的内容提取
    for tool_idx in range(len(boundaries)):
        name, args = parser._get_current_tool_content(content, tool_idx)
        print(f"Tool {tool_idx}: name={name}, args={args}")
