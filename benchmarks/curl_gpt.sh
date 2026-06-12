MODEL="${MODEL:-/data/models/gpt-oss-120b-w-mxfp4-a-fp8}" \
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": ${MODEL},
  "prompt": "What is the capital of France, and what is it known for?",
  "temperature": 0.0,
  "max_tokens": 100
}'
