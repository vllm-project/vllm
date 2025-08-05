from vllm.entrypoints.openai.tool_parsers.minimax_tool_parser import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# 创建parser
tokenizer = get_tokenizer('MiniMaxAi/MiniMax-M1-40k')
parser = MinimaxToolParser(tokenizer)

# 重置状态
parser._reset_streaming_state()

# 模拟逐字符流式处理
test_text = '''<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}
{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}
</tool_calls>'''

print(f"Test text: {repr(test_text)}")
print(f"Length: {len(test_text)}")
print()

tool_calls_detected = []

# 关键节点测试
key_points = [
    len('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}'),
    len('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n'),
    len('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n{"name": "calculate_area", "arguments": {"shape": "rectangle", "dimensions": {"width": 10, "height": 5}}}'),
    len(test_text)
]

for point in key_points:
    current_text = test_text[:point]
    previous_text = test_text[:point-1] if point > 0 else ""
    delta_text = test_text[point-1:point] if point > 0 else test_text[0:1]
    
    print(f"\n=== Testing at position {point} ===")
    print(f"Current text ends with: {repr(current_text[-20:])}")
    print(f"Delta: {repr(delta_text)}")
    
    result = parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )
    
    print(f"Streaming state: {parser.streaming_state}")
    
    if result and hasattr(result, 'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            tool_info = {
                'position': point,
                'function_name': tool_call.function.name if tool_call.function else None,
                'arguments': tool_call.function.arguments if tool_call.function else None,
            }
            tool_calls_detected.append(tool_info)
            print(f"  Tool call: {tool_call.function.name} - {tool_call.function.arguments}")

print(f"\n=== Final Results ===")
print(f"Total tools detected: {len(tool_calls_detected)}")
for tc in tool_calls_detected:
    print(f"  {tc}")
