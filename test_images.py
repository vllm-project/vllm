import requests
import json

url = "http://127.0.0.1:8000/v1/chat/completions"

def test():

    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": '/infra/chp/vlm_opt/0001.png'
                        },
                    },
                    {
                        "type": "text",
                        "text": "描述提供的图片",
                    }
                ],
            }
        ],
        "tools": [],
        "stream": False,
        "do_sample": True,
        "best_of": 1,
        "top_p": 0.95,
        "top_k": 1,
        "max_token": 1024,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
        "decoder_input_details": False,
        "enable_thinking": True,
        "wrap_server_preprocess": True,
        "metadata": {
            "video_sample_config": {
                "fps_interval": 1,
                "max_tokens_per_frame": 768,
                "maximum_frame_count": 640
            }
        }
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    if response.status_code == 200:
        if data["stream"]:
            final_output = ""
            for line in response.iter_lines():
                decoded_line = line.decode('utf-8')

                if not decoded_line:
                    continue

                if decoded_line.startswith("data: "):
                    json_string = decoded_line[6:]
                    if json_string.strip() == "[DONE]":
                        break

                    data_obj = json.loads(json_string)

                    if 'choices' in data_obj and data_obj['choices']:
                        delta_content = data_obj['choices'][0].get('delta', {}).get('content', '')
                        final_output += delta_content
                        # 如果想实时打印，可以取消下面这行注释
                        # print(delta_content, end='', flush=True)

            print(final_output)
            return 0, final_output
        else:
            output = response.json()['choices'][0]['message']['content']
            print(response.json())
            print(output)
            return 0, output
    else:
        print(f"Received bad status code: {response.status_code}")
        return -1, ""

if __name__ == '__main__':
    test()