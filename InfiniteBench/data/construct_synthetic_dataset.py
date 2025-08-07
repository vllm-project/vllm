import jsonlines
import random 
import os
import re
import importlib.util
import json 


def build_number_string():
    #####32
    # prompt = "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n"
    #####25
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    #####26
    ans = "The sequence of digits is {key}. Remember it. {key} is the sequence of digits.\n"
    #####10
    question = "What is the sequence of digits?"


    target_length = [1024 * 64, 1024 * 128]
    num_noise = [2610, 5220]
    step = [45, 90]
    repeat_time = 10
    for i in range(1, 2):
        target_length_i = target_length[i]
        step_i = step[i]
        num_noise_i = num_noise[i]
        ret = []
        for j in range(0, num_noise_i+1, step_i):
            input_text =  noise * j + ans + noise * (num_noise_i - j)
            for t in range(repeat_time):
                keys = []
                for k in range(5):
                    keys.append(str(random.randint(0,9)))
                for k in range(5):
                    pos = random.randint(0,5+k-1)
                    keys.insert(pos, keys[pos])
                key_t = "".join(keys)
                ret.append({"context": input_text.replace("{key}", key_t), "answer": key_t, "input": question, "len": 26 * (num_noise_i - j)})
        fw = jsonlines.open("number_string.jsonl", 'w')
        fw.write_all(ret)
        fw.close()


def build_passkey():
    #####32
    # prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
    #####25
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    #####26
    ans = "The pass key is {key}. Remember it. {key} is the pass key.\n"
    #####10
    question = "What is the pass key?"

    target_length = [1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128, 1024 * 256]
    num_noise = [326, 652, 1305, 2610, 5220, 10440]
    step = [6,12 ,22, 45, 90, 180]
    repeat_time = 5
    for i in range(0,4):
        target_length_i = target_length[i]
        step_i = step[i]
        num_noise_i = num_noise[i]
        ret = []
        for j in range(0, num_noise_i+1, step_i):
            input_text = noise * j + ans + noise * (num_noise_i - j)
            for t in range(repeat_time):
                keys = []
                for k in range(5):
                    keys.append(str(random.randint(0,9)))
               
                key_t = "".join(keys)
                ret.append({"input": question, "context": input_text.replace("{key}", key_t), "answer": key_t, "len": 26 * (num_noise_i - j)})
        fw = jsonlines.open("passkey_%d.jsonl"%target_length_i, 'w')
        fw.write_all(ret)
        fw.close()


def build_kv_retrieval():

    target_length = [64 * 1024, 128 * 1024]
    # interv = [16, 7]
    nsample = [500, 500]
    nnoise = [928, 2500]
    for ii in range(1, 2):
        cnt = -1
        ret = []

        with jsonlines.open("kv-retrieval-3000_keys.jsonl") as fin:
            for line in fin:
                print(len(line["ordered_kv_records"]))
                # return 0
                cnt += 1
                if cnt == nsample[ii]:
                    break
                ans_id = min(int(cnt * nnoise[ii] / nsample[ii]), nnoise[ii])

                text = "JSON data:\n{"
                t = -1
                random.shuffle(line["ordered_kv_records"])
                for item in line["ordered_kv_records"]:
                    t += 1
                    if t == nnoise[ii]:
                        break
                    text += "\"" + item[0] + "\": \"" + item[1] + "\", "
                text = text[:-2] + '}'
                question = "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                # text += "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                # print(len(tokenizer.encode(text)))
                # break
                ret.append({"id": cnt, "context": text, "input": question, "answer": line["ordered_kv_records"][ans_id][1]})
            
        
        fw = jsonlines.open("kv_retrieval.jsonl", 'w')
        fw.write_all(ret)
        fw.close()


def generate_random_list(length, _min, _max, task):
    # random_list = [random.randint(_min, _max) for _ in range(length)]
    # ret_list = random_list.copy()
    
    if task == "largest number":
        _max = random.randint(int(_max * 0.8), _max)
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        ans = max(random_list)
        input = str(ret_list)
    elif task == "second largest number":
        _max = random.randint(int(_max * 0.8), _max)
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        target = max(random_list)
        while target == max(random_list):
            random_list.remove(max(random_list))
        ans = max(random_list)
        input = str(ret_list)

    elif task == "third largest number":
        _max = random.randint(int(_max * 0.8), _max)
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        target = max(random_list)
        while target == max(random_list):
            random_list.remove(max(random_list))
        target = max(random_list)
        while target == max(random_list):
            random_list.remove(max(random_list))
        ans = max(random_list)
        input = str(ret_list)
    
    elif task == "smallest number":
        _min = random.randint(_min, int(_max * 0.2))
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        ans = min(random_list)
        input = str(ret_list)
    
    elif task == "second smallest number":
        _min = random.randint(_min, int(_max * 0.2))
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        target = min(random_list)
        while target == min(random_list):
            random_list.remove(min(random_list))
        ans = min(random_list)
        input = str(ret_list)

    elif task == "third smallest number":
        _min = random.randint(_min, int(_max * 0.2))
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        target = min(random_list)
        while target == min(random_list):
            random_list.remove(min(random_list))
        target = min(random_list)
        while target == min(random_list):
            random_list.remove(min(random_list))
        ans = min(random_list)
        input = str(ret_list)
    elif task == "median":
        if random.random() > 0.5:
            _min = random.randint(_min, int(_max * 0.2))
            random_list = [random.randint(_min, _max) for _ in range(length)]
        else:
            _max = random.randint(int(_max * 0.8), _max)
            random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        random_list.sort()
        if len(random_list)%2 == 1:
            ans = random_list[len(random_list)//2]
        else:
            ans = (random_list[len(random_list)//2] + random_list[len(random_list)//2-1])/2
        input = str(ret_list)
    elif task ==  "expression":
        random_list = [random.randint(_min, _max) for _ in range(length)]
        ret_list = random_list.copy()
        input = str(random_list[0])
        value = random_list[0]
        ans = []
        for i in range(1, length):
            poss = random.random()
            if poss > 0.5:
                if value + random_list[i] > _max:
                    random_list[i] = random.randint(_min, _max-value)

                input += " + " + str(random_list[i])
                value += random_list[i]
              
            else:
                if value - random_list[i] < 0:
                    random_list[i] = random.randint(_min, value)
                input += " - " + str(random_list[i])
                value -= random_list[i]
            ans.append(value)

            
    else:
        print("Invalid task")
        ans = None

    return ans, input


def generate_math_qa(list_length, min_val, max_val, tasks=None):
    num_samples = 50
    ret = []
    prompts = {
        "largest number": "Find the largest number from the list below:",
        "second largest number": "Find the second largest number from the list below:",
        "third largest number": "Find the third largest number from the list below:",
        "smallest number": "Find the smallest number from the list below:",
        "second smallest number": "Find the second smallest number from the list below:",
        "third smallest number": "Find the third smallest number from the list below:",
        "median": "Calculate the median number from the list below:",
        "expression": "Calculate the numerical expression and provide intermediate results only, for example, for the expression 1 + 3 + 10 - 8, output 4, 14, 6 without displaying the steps.\n\nCalculate the value of the expression below:",
    }
    inputs = {
        "largest number": "You should answer with only one number, no other words. The largest number of the list is: ",
        "second largest number": "You should answer with only one number, no other words. The second largest number of the list is: ",
        "third largest number": "You should answer with only one number, no other words. The third largest number of the list is: ",
        "smallest number": "You should answer with only one number, no other words. The smallest number of the list is: ",
        "second smallest number": "You should answer with only one number, no other words. The second smallest number of the list is: ",
        "third smallest number": "You should answer with only one number, no other words. The third smallest number of the list is: ",
        "median": "You should answer with only one number, no other words. The median number of the list is: ",
        "expression": "The value of the numerical expression is: ",
    }
    for i in range(len(tasks)):
        for _ in range(num_samples):
            std_out, context = generate_random_list(list_length, min_val, max_val, tasks[i])

            ret.append({"prompt": prompts[tasks[i]], "context": context, "input": inputs[tasks[i]], "answer": std_out})
    return ret


def build_math_find():
    list_length = 60000  # Length of the generated lists

    min_val = 0  # Minimum value for list elements
    max_val = 99  # Maximum value for list elements

    ret = generate_math_qa(list_length, min_val, max_val, tasks=["largest number", "second largest number", "third largest number", "smallest number", "second smallest number", "third smallest number", "median"])

    # Save the data to a JSONL file
    fw = jsonlines.open("math_find.jsonl", "w")
    fw.write_all(ret)
    fw.close()


def build_math_calc():
    list_length = 30000  # Length of the generated lists

    min_val = 0  # Minimum value for list elements
    max_val = 99  # Maximum value for list elements

    ret = generate_math_qa(list_length, min_val, max_val, tasks=["expression"])

    # Save the data to a JSONL file
    fw = jsonlines.open("math_calc.jsonl", "w")
    fw.write_all(ret)
    fw.close()


def generate_and_store_collections(n, m, min_val, max_val, output_file):
    total_elements = n * m
    collection = set()

    while len(collection) < total_elements:
        collection.add(random.randint(min_val, max_val))
    
    collection = list(collection)
    random.shuffle(collection)

    collections = [collection[i * m: (i + 1) * m] for i in range(n)]

    with open(output_file, 'w') as file:
        json.dump(collections, file)


def generate_functions(input_file, min_add, max_add, output_file):
    with open(input_file, 'r') as file:
        collections = json.load(file)
    
    function_list = []

    for i in range(len(collections)):
        for t in collections[i]:
            function = f"def func_{t}(x):\n"
            if i < len(collections) - 1:
                next_collection = collections[i + 1]
                k = random.choice(next_collection)
                addition = random.randint(min_add, max_add)
                if addition == 0:
                    function += f"    return func_{k}(x)\n"
                elif addition < 0:
                    function += f"    return func_{k}(x) - {-addition}\n"
                else:
                    function += f"    return func_{k}(x) + {addition}\n"
            else:
                addition = random.randint(min_add, max_add)
                if addition == 0:
                    function += f"    return x\n"
                elif addition < 0:
                    function += f"    return x - {-addition}\n"
                else:
                    function += f"    return x + {addition}\n"
            function_list.append((f"func_{t}", function))

    function_list.sort(key=lambda x: int(x[0].split("_")[1]))

    with open(output_file, 'w') as out:
        for _, func_text in function_list:
            out.write(func_text)
            out.write("\n")
    

def generate_code_run_example(collection_file, min_x, max_x, functions_module, functions_file='functions_module.py'):
    spec = importlib.util.spec_from_file_location("functions_module", functions_module)
    functions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(functions)
    # print(functions)
    # load all functions in functions_module.py and store them in a string
    content = f"\nHere is the content of {functions_file}:\n\n"
    with open(functions_module, 'r') as file:
        for line in file:
            content += line

    with open(collection_file, 'r') as file:
        collections = json.load(file)


    j = random.choice(collections[0])
    x = random.randint(min_x, max_x)
    test_sample = {
            "context": content,
            "answer": getattr(functions, f"func_{j}")(x),
            "input": f"Please give me the exact number of the return value of func_{j}({x}). Your response should end with the sentence 'The return value is:'.",
    }

    return test_sample
    # with jsonlines.open(output_file_samples, mode='w') as writer:
    #     writer.write_all(test_samples)
    # with jsonlines.open(output_file_answers, mode='w') as writer:
    #     writer.write_all(test_answers)



def build_code_run():
    MAX_NUM_FUNC = 550
    min_val = 1 # minimum value of function indeces
    max_val = 2*MAX_NUM_FUNC # maximum value of function indeces
    max_add = 17 # maximum value of addition in return expression
    min_add = -12 # minimum value of addition in return expression
    collections_file = 'collections.json'
    functions_file = 'functions_module.py'
    #------------------------------------------------------------------------#
    # Parameters for generating test samples and answers
    num_test = 1
    min_x = -10
    max_x = 10
    n_list = [2, 4, 6, 8, 10]
    ret = []
    cnt = -1
    for i in range(len(n_list)):
        for _ in range(80):
            cnt += 1
            while True:
                try:
                    generate_and_store_collections(n_list[i], int(MAX_NUM_FUNC/n_list[i]), min_val, max_val, collections_file)
        
                    generate_functions(collections_file, min_add, max_add, functions_file)

                    example = generate_code_run_example(collections_file, min_x, max_x, functions_file)
                    example['id'] = cnt
            
                    ret.append(example)
                    break
                except Exception as e:
                    print(e)
    fw = jsonlines.open("code_run.jsonl", 'w')
    fw.write_all(ret)
    fw.close()

if __name__ == "__main__":
    # os.system("git clone https://github.com/nelson-liu/lost-in-the-middle.git")
    # os.system("python3.10 -u lost-in-the-middle/scripts/make_kv_retrieval_data.py --num-keys 3000 --num-examples 500 --output-path kv-retrieval-3000_keys.jsonl.gz")
    # os.system("gzip -d kv-retrieval-3000_keys.jsonl.gz")
    # build_kv_retrieval()
    # build_passkey()
    # build_number_string()
    # build_math_find()
    # build_math_calc()
    build_code_run()

