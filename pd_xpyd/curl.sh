curl http://127.0.0.1:8868/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/",
        "prompt": "请用中文思考和回答：A paper punch can be placed at any point in the plane, and when it operates, it can punch out points at an irrational distance from it. What is the minimum number of paper punches needed to punch out all points in the plane?",
        "max_tokens": 1000,
        "temperature": 0
    }'

