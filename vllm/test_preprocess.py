
def llama2_preprocess(prompt):
    prompt_start = "<s>[INST]"
    prompt_str = "[/INST]"
    for i, p in enumerate(reversed(prompt)):
        if p['role'] == 'user':
            content = '<s>' + p['content'] + "\n</s>"
        else:
            content = '<s>' + p['content'] + "\n</s>"
        prompt_str = content + prompt_str
    prompt_str = prompt_start + prompt_str
    return prompt_str

if __name__ == "__main__":
    # prompt_str = "你是私有数据库搜索引擎，正在根据用户的问题搜索和总结答案，请认真作答，并给出出处。\n            用户的问题为：\n            哪些文件里面有盐雾的测试条件？\n            你搜索到的相关内容为：\n            HQ WD6931 手表主板DFM 报告_20240102/slide_5 ：文件标题：\nHQ WD6931 手表主板DFM 报告_20240102/slide_5\n:/n文件内容：# WD6931手表主板防护要求\n\n__待防护产品__\n\n__    __ WD6931手表主板尺寸: 36\\.7 x 30\\.8 mm\n\n__UPH __  __要求__\n\nTBD\n\n__防护要求__\n\n需求一：\n\n1\\. 整机测试要求：120H高温高湿（55℃，95%RH）开机存储测试120H检查外观、功能\n\n2\\. 板级测试：裸板IPX4\n\n需求二：\n\n1\\. 整机测试要求：120H高温高湿（55℃，95%RH）开机存储测试120H检查外观、功能\n\n2\\. 板级测试: 裸板通电状态下喷淋盐水10s，放入温箱80度保持10mins后，检查电流是否正常。共重复做20个循环。\n\n\nD手机镀膜 DFM报告/pdf_13 ：文件标题：\nD手机镀膜 DFM报告/pdf_13\n:/n文件内容：IQC摆盘干燥\nOQC 检测 纳米镀膜\n镀膜工序流程\n\n"
    prompt_str = "请回答问题"
    prompt = [{"role": "user", "content": prompt_str}]
    prompt_str = llama2_preprocess(prompt)
    print(prompt_str)