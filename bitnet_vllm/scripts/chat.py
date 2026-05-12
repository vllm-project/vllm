"""Interactive chat client for BitNet served via vLLM OpenAI API.

No external dependencies — uses only Python stdlib.
"""
import json
import sys
import urllib.request
import urllib.error


API_BASE = "http://localhost:8000/v1"
MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"


def api_post(endpoint, data, timeout=600):
    """POST JSON to the API and return parsed response.
    
    First request may take several minutes due to CUDA graph compilation.
    """
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def text_completion(prompt, max_tokens=256, temperature=0.7):
    """Send a text completion request."""
    result = api_post("completions", {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    return result["choices"][0]["text"]


def check_server():
    """Check if the server is running."""
    try:
        req = urllib.request.Request(f"{API_BASE}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read().decode("utf-8"))["data"]
            print(f"  Server is running. Model: {models[0]['id']}")
            return True
    except urllib.error.URLError:
        print("  Cannot connect to server at localhost:8000")
        print("  Start it first with: .\\docker-bitnet.ps1 -Serve")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("  BitNet b1.58 — Interactive Text Completion")
    print("  (powered by vLLM)")
    print("=" * 60)
    print()

    if not check_server():
        sys.exit(1)

    print()
    print("  Note: BitNet is a base model (not chat-tuned).")
    print("  It completes text, not conversations.")
    print()
    print("  Type a prompt and press Enter to generate.")
    print("  Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            print("\rGenerating (first request may take 2-3 min for warmup)...", end="", flush=True)
            reply = text_completion(user_input)
            # Clear the generating message and print result
            print(f"\r{' ' * 70}\r{user_input}{reply}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"\rError {e.code}: {body[:200]}")
        except Exception as e:
            print(f"\rError: {e}")


if __name__ == "__main__":
    main()
