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
# Limitations (enforced server-side):
#   - stream=true is not supported with batched messages
#   - tools / tool_choice are not supported with batched messages
#   - beam_search (use_beam_search) is not supported with batched messages

BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# ---------------------------------------------------------------------------
# 1. Simple batch – two plain-text conversations
# ---------------------------------------------------------------------------
echo "=== Example 1: simple batch ==="
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
      [{\"role\": \"user\", \"content\": \"What is the capital of Japan?\"}]
    ]
  }" | jq '.choices[] | {index, content: .message.content}'

# ---------------------------------------------------------------------------
# 2. Batch with regex-constrained structured output
# ---------------------------------------------------------------------------
echo ""
echo "=== Example 2: batch with regex constraint (yes|no) ==="
curl -s "${BASE_URL}/v1/chat/completions" \
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
echo "=== Example 3: batch with json_schema ==="
curl -s "${BASE_URL}/v1/chat/completions" \
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
# 4. Batch with a rich JSON schema – company description summaries
#    (two companies in one request)
# ---------------------------------------------------------------------------
echo ""
echo "=== Example 4: batch company description summaries ==="
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      [
        {\"role\": \"system\", \"content\": \"You are a business analyst. Extract structured information from company descriptions.\"},
        {\"role\": \"user\",   \"content\": \"Summarize: Acme Corp designs and manufactures road runner traps. Founded in 1949 in Arizona, they serve the cartoon character market with explosive devices and elaborate contraptions. Their revenue is $42M.\"}
      ],
      [
        {\"role\": \"system\", \"content\": \"You are a business analyst. Extract structured information from company descriptions.\"},
        {\"role\": \"user\",   \"content\": \"Summarize: Initech is a software consultancy founded in 1993 in Texas. They develop enterprise resource planning solutions for mid-size companies. Revenue $15M.\"}
      ]
    ],
    \"response_format\": {
      \"type\": \"json_schema\",
      \"json_schema\": {
        \"name\": \"company_summary\",
        \"strict\": true,
        \"schema\": {
          \"type\": \"object\",
          \"properties\": {
            \"short\": {
              \"type\": \"string\",
              \"description\": \"A one-sentence description of the company\"
            },
            \"industry\": {
              \"type\": \"string\",
              \"description\": \"Primary industry or sector\"
            },
            \"founded_year\": {
              \"type\": \"integer\",
              \"description\": \"Year the company was founded\"
            },
            \"hq_country\": {
              \"type\": \"string\",
              \"description\": \"Country where the headquarters is located\"
            }
          },
          \"required\": [\"short\", \"industry\", \"founded_year\", \"hq_country\"]
        }
      }
    }
  }" | jq '.choices[] | {index, summary: (.message.content | fromjson)}'
