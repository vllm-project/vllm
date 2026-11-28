import json
import random

def generate_dataset(file_path):
    # Requirements:
    # 70-80 total requests (we'll do 75)
    # 25 heavy requests (prefill ~500k tokens)
    # 50 short apps (prefill ~1k tokens)
    
    num_heavy = 25
    num_short = 50
    
    # 500k tokens is quite large; we'll generate lists of 500,000 ints.
    # To keep file size manageable and generation fast, we can use a repeating list.
    base_heavy_tokens = [1] * 500000
    base_short_tokens = [1] * 1000
    
    requests = []
    
    for i in range(num_heavy):
        requests.append({
            "prompt": "Heavy request " + str(i),
            "prompt_token_ids": base_heavy_tokens,
            "output_token_ids": [2] * 128
        })
        
    for i in range(num_short):
        requests.append({
            "prompt": "Short request " + str(i),
            "prompt_token_ids": base_short_tokens,
            "output_token_ids": [2] * 128
        })
        
    random.shuffle(requests)
    
    with open(file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
            
if __name__ == "__main__":
    generate_dataset("unicaja_dataset.jsonl")
    print("Dataset generated at unicaja_dataset.jsonl")
