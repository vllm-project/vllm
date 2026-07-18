# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""将 Anthropic Messages 请求体转换为 OpenAI Chat Completions 请求体。

用法:
    python anthropic_to_openai.py claude_code_request.json openai_request.json
"""

import argparse
import json


def convert_tools(anthropic_tools):
    """Anthropic tools -> OpenAI tools (function 形式)。"""
    openai_tools = []
    for t in anthropic_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def convert_tool_choice(tc):
    """Anthropic tool_choice -> OpenAI tool_choice。"""
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


def blocks_to_text(content):
    """把 content(字符串或 block 列表)里的 text 块合并成纯文本。"""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if b.get("type") == "text":
            parts.append(b["text"])
    return "".join(parts)


def convert_messages(anthropic_messages, system):
    """Anthropic messages(+顶层 system) -> OpenAI messages。"""
    openai_messages = []

    # 顶层 system 转为一条 system 消息
    if system:
        openai_messages.append({"role": "system", "content": blocks_to_text(system)})

    for m in anthropic_messages:
        role = m["role"]
        content = m["content"]

        # 纯字符串内容,直接透传
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for b in content:
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b["text"])
                elif bt == "thinking":
                    # OpenAI 输入不接受 reasoning/thinking,丢弃
                    continue
                elif bt == "tool_use":
                    tool_calls.append(
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(
                                    b.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            msg = {"role": "assistant"}
            msg["content"] = "".join(text_parts) if text_parts else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            openai_messages.append(msg)

        else:  # user
            # tool_result 块要拆成独立的 role=tool 消息
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            if tool_results:
                for b in tool_results:
                    tc = b.get("content", "")
                    # tool_result 的 content 可能是字符串或 block 列表
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": blocks_to_text(tc)
                            if not isinstance(tc, str)
                            else tc,
                        }
                    )
            else:
                # 普通 user 文本(合并多个 text 块;去掉 cache_control 等)
                openai_messages.append(
                    {"role": "user", "content": blocks_to_text(content)}
                )

    return openai_messages


def convert(anthropic_req):
    out = {
        "model": anthropic_req["model"],
        "messages": convert_messages(
            anthropic_req.get("messages", []), anthropic_req.get("system")
        ),
        "stream": anthropic_req.get("stream", False),
    }

    # 采样参数透传
    if "temperature" in anthropic_req:
        out["temperature"] = anthropic_req["temperature"]
    if "max_tokens" in anthropic_req:
        out["max_tokens"] = anthropic_req["max_tokens"]
    # vLLM 支持的扩展采样参数
    if "skip_special_tokens" in anthropic_req:
        out["skip_special_tokens"] = anthropic_req["skip_special_tokens"]

    # tools / tool_choice
    if anthropic_req.get("tools"):
        out["tools"] = convert_tools(anthropic_req["tools"])
    tc = convert_tool_choice(anthropic_req.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic Messages -> OpenAI Chat Completions 转换器"
    )
    parser.add_argument("input", help="Anthropic 请求体 JSON 文件")
    parser.add_argument(
        "output",
        nargs="?",
        default="openai_request.json",
        help="输出的 OpenAI 请求体 JSON 文件(默认 openai_request.json)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        anthropic_req = json.load(f)

    openai_req = convert(anthropic_req)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(openai_req, f, ensure_ascii=False, indent=2)

    print(f"已写出 OpenAI 请求体 -> {args.output}")
    print(
        f"消息数: {len(openai_req['messages'])}, "
        f"工具数: {len(openai_req.get('tools', []))}"
    )


if __name__ == "__main__":
    main()
