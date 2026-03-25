#!/usr/bin/env bash
# Examples of batched chat completions via the vLLM OpenAI-compatible API.
#
# The `messages` field accepts either a single conversation (standard) or a
# list of conversations (batched).  In batch mode each conversation becomes
# one choice in the response, indexed 0, 1, …, N-1.
#
# Start a server first, e.g.:
#   vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
#
# Current limitations:
#   - stream=true is not supported with batched messages
#   - tools / tool_choice are not supported with batched messages
#   - beam_search (use_beam_search) is not supported with batched messages

BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# ---------------------------------------------------------------------------
# 1. Simple batch, one or two plain-text conversations.
# ---------------------------------------------------------------------------
echo " Example 1a: simple batch with 1 request "
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is the capital of Japan?\"}
    ]
  }" | jq '.choices[] | {index, content: .message.content}'

echo " Example 1b: simple batch with 2 requests "
curl -s "${BASE_URL}/v1/chat/completions/batch" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
      [{\"role\": \"user\", \"content\": \"What is the capital of Japan?\"}]
    ]
  }" | jq '.choices[] | {index, content: .message.content}'

# ---------------------------------------------------------------------------
# 2. Batch with regex-constrained structured output.
# ---------------------------------------------------------------------------
echo ""
echo " Example 2: batch with regex constraint (yes|no) "
curl -s "${BASE_URL}/v1/chat/completions/batch" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [{\"role\": \"user\", \"content\": \"Is the sky blue? Answer yes or no.\"}],
      [{\"role\": \"user\", \"content\": \"Is fire cold? Answer yes or no.\"}]
    ],
    \"extra_body\": {
      \"structured_outputs\": {\"regex\": \"(yes|no)\"}
    }
  }" | jq '.choices[] | {index, answer: .message.content}'

# ---------------------------------------------------------------------------
# 3. Batch with JSON schema structured output
# ---------------------------------------------------------------------------
echo ""
echo " Example 3: batch with json_schema "
curl -s "${BASE_URL}/v1/chat/completions/batch" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [{\"role\": \"user\", \"content\": \"Describe the person: name Alice, age 30.\"}],
      [{\"role\": \"user\", \"content\": \"Describe the person: name Bob, age 25.\"}]
    ],
    \"response_format\": {
      \"type\": \"json_schema\",
      \"json_schema\": {
        \"name\": \"person\",
        \"strict\": true,
        \"schema\": {
          \"type\": \"object\",
          \"properties\": {
            \"name\": {\"type\": \"string\", \"description\": \"Full name of the person\"},
            \"age\": {\"type\": \"integer\", \"description\": \"Age in years\"}
          },
          \"required\": [\"name\", \"age\"]
        }
      }
    }
  }" | jq '.choices[] | {index, person: (.message.content | fromjson)}'

# ---------------------------------------------------------------------------
# 4. Batch with a rich JSON schema – book summaries
#    (two books in one request, extracting structured metadata).
# ---------------------------------------------------------------------------
echo ""
echo " Example 4: batch book summaries "
curl -s "${BASE_URL}/v1/chat/completions/batch" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [
        {\"role\": \"system\", \"content\": \"You are a literary analyst. Extract structured information from book descriptions.\"},
        {\"role\": \"user\",   \"content\": \"Extract information from this book: '1984' by George Orwell, published in 1949, 328 pages. A dystopian novel set in a totalitarian society ruled by Big Brother, following Winston Smith as he secretly rebels against the oppressive Party that surveils and controls every aspect of life.\"}
      ],
      [
        {\"role\": \"system\", \"content\": \"You are a literary analyst. Extract structured information from book descriptions.\"},
        {\"role\": \"user\",   \"content\": \"Extract information from this book: 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams, published in 1979, 193 pages. A comedic science fiction novel following Arthur Dent, an ordinary Englishman who is whisked off Earth moments before it is demolished to make way for a hyperspace bypass, and his subsequent absurd adventures across the universe.\"}
      ]
    ],
    \"response_format\": {
      \"type\": \"json_schema\",
      \"json_schema\": {
        \"name\": \"book_summary\",
        \"strict\": true,
        \"schema\": {
          \"type\": \"object\",
          \"properties\": {
            \"author\": {
              \"type\": \"string\",
              \"description\": \"Full name of the author\"
            },
            \"num_pages\": {
              \"type\": \"integer\",
              \"description\": \"Number of pages in the book\"
            },
            \"short_summary\": {
              \"type\": \"string\",
              \"description\": \"A one-sentence summary of the book\"
            },
            \"long_summary\": {
              \"type\": \"string\",
              \"description\": \"A detailed two to three sentence summary covering the main themes and plot\"
            }
          },
          \"required\": [\"author\", \"num_pages\", \"short_summary\", \"long_summary\"]
        }
      }
    }
  }" | jq '.choices[] | {index, book: (.message.content | fromjson)}'
