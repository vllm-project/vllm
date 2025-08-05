from vllm.entrypoints.openai.tool_parsers.minimax_tool_parser import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# 创建parser
tokenizer = get_tokenizer('MiniMaxAi/MiniMax-M1-40k')
parser = MinimaxToolParser(tokenizer)

# 重置状态
parser._reset_streaming_state()

# 测试文本
test_text = '<tool_calls>\n{"name": "tool1", "arguments": {"a": 1}}\n{"name": "tool2", "arguments": {"b": 2}}\n</tool_calls>'

print(f"Test text: {repr(test_text)}")
print()

# 找到第二个工具完整出现的位置
second_tool_complete = test_text.find('{"name": "tool2", "arguments": {"b": 2}}') + len('{"name": "tool2", "arguments": {"b": 2}}')
print(f"Second tool complete at position: {second_tool_complete}")

# 测试关键位置
key_positions = [
    54,  # 第一个工具完成，current_idx变成1
    55,  # 第二个工具开始
    75,  # 第二个工具的名称出现
    second_tool_complete,  # 第二个工具完整
    len(test_text)  # 结束
]

for pos in key_positions:
    print(f"\n=== Position {pos} ===")
    current_text = test_text[:pos]
    previous_text = test_text[:pos-1] if pos > 0 else ""
    delta_text = test_text[pos-1:pos] if pos > 0 else test_text[0:1]
    
    print(f"Current text ends: {repr(current_text[-30:])}")
    print(f"Delta: {repr(delta_text)}")
    
    # 手动检查工具检测
    tool_content_start = current_text.find('<tool_calls>') + len('<tool_calls>')
    tool_content_end = current_text.find('</tool_calls>')
    if tool_content_end == -1:
        tool_content = current_text[tool_content_start:]
    else:
        tool_content = current_text[tool_content_start:tool_content_end]
    
    boundaries = parser._find_tool_boundaries(tool_content)
    tool_count = parser._detect_tools_in_text(tool_content)
    
    print(f"Tool content: {repr(tool_content)}")
    print(f"Boundaries: {boundaries}")
    print(f"Tool count: {tool_count}")
    print(f"Current state: current_idx={parser.streaming_state['current_tool_index']}")
    
    # 调用流式处理
    result = parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )
    
    print(f"New state: current_idx={parser.streaming_state['current_tool_index']}")
    
    if result and hasattr(result, 'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            print(f"  Tool call: {tool_call.function.name} - {tool_call.function.arguments}")
