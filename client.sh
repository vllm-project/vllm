export host="0.0.0.0"
export port="12301"
export prompt="你叫什么名字？"
#Llama2-13B
#Baichuan2-13B
#Mixtral-8x7B-v0.1
python3 ./examples/api_client_switch.py --host 0.0.0.0 --port 12304 --modeltype Baichuan2-13B
python3 ./examples/api_client.py --host $host --port $port --prompt $prompt --stream