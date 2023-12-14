from vllm_wrapper import vLLMWrapper

if __name__ == '__main__':
    model = ""
    #gptq a16w4 model
    vllm_model = vLLMWrapper(model,
                             quantization = 'gptq',
                             dtype="float16",
                             tensor_parallel_size=2)
    #a16w16 model
    # vllm_model = vLLMWrapper(model,
    #                          dtype="bfloat16",
    #                          tensor_parallel_size=2)

    response, history = vllm_model.chat(query="你好",
                                        history=None)
    print(response)
    response, history = vllm_model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。",
                                        history=history)
    print(response)
    response, history = vllm_model.chat(query="给这个故事起一个标题",
                                        history=history)
    print(response)

