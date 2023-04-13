import os


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    rates = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    duration = 1000
    for rate in rates:
        cmd = f"python benchmark/benchmark_chatbot.py --dataset sharegpt_clean_lang_10k_opt_tokenized.pkl --model facebook/opt-13b --request-rate {rate} --duration {duration} --n1 1.0 --use-dummy"
        run_cmd(cmd)
