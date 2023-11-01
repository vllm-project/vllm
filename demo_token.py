from transformers import AutoTokenizer, AutoModel


def demo():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    # xx = tokenizer("给六岁小朋友解释一下万有引")
    # print(xx)
    zz = [64790, 64792, 30910, 54840, 55139, 55201, 34408, 32929, 32024, 54809,
          54536, 54976]
    zz = [32365, 13581, 283, 1934, 729, 6631, 30954, 449, 30917, 2]
    zz = [32365, 32885, 13, 54809, 54536, 51635, 54532, 44651, 31697, 31623,
          31845, 32885, 30932, 54964, 39741, 31754]
    for z in zz:
        response = tokenizer.decode([z])
        print(response, len(response))


def run_hf():
    prompt = "给六岁小朋友解释一下万有引"
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # print("111111111", input_ids)
    max_tokens = 16
    output_ids = model.generate(
        input_ids.cuda(),
        use_cache=True,
        do_sample=False,
        max_new_tokens=max_tokens,
    )
    # print("222222222", output_ids)

    output_str = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_str)
    del model


def run_llm(checkpoint):
    from vllm import LLM, SamplingParams
    llm = LLM(model=checkpoint, trust_remote_code=True, max_model_len=8192)
    prompt = "给六岁小朋友解释一下万有引"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = llm.generate(prompt, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(output)
    del llm


if __name__ == '__main__':
    # demo()
    # run_hf()
    # print("-" * 100)
    # run_llm("THUDM/chatglm2-6b")
    print("-" * 100)
    run_llm("THUDM/chatglm3-6b")
