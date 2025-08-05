from vllm.entrypoints.openai.tool_parsers.minimax_tool_parser import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# 创建parser
tokenizer = get_tokenizer('MiniMaxAi/MiniMax-M1-40k')
parser = MinimaxToolParser(tokenizer)

# 重置状态
parser._reset_streaming_state()

# 简化的测试文本
test_text = '''<tool_calls>
{"name": "tool1", "arguments": {"a": 1}}
{"name": "tool2", "arguments": {"b": 2}}
</tool_calls>'''

print(f"Test text: {repr(test_text)}")
print()

tool_calls_detected = []

# 关键区间测试
key_ranges = [
    (0, 50),   # 第一个工具的名称和开始
    (50, 75),  # 第一个工具完成到第二个工具开始
    (75, len(test_text))  # 第二个工具完成
]

for start, end in key_ranges:
    print(f"\n=== Testing range {start} to {end} ===")
    
    for i in range(start, min(end, len(test_text))):
        current_text = test_text[:i+1]
        previous_text = test_text[:i] if i > 0 else ""
        delta_text = test_text[i:i+1]
        
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        
        if result and hasattr(result, 'tool_calls') and result.tool_calls:
            for tool_call in result.tool_calls:
                tool_info = {
                    'position': i+1,
                    'char': repr(delta_text),
                    'function_name': tool_call.function.name if tool_call.function else None,
                    'arguments': tool_call.function.arguments if tool_call.function else None,
                }
                tool_calls_detected.append(tool_info)
                print(f"  Pos {i+1} ({repr(delta_text)}): {tool_call.function.name} - {tool_call.function.arguments}")
                print(f"    State: current_idx={parser.streaming_state['current_tool_index']}")

print(f"\n=== Final Results ===")
print(f"Total tools detected: {len(tool_calls_detected)}")
for tc in tool_calls_detected:
    print(f"  Pos {tc['position']}: {tc['function_name']} - {tc['arguments']}")
