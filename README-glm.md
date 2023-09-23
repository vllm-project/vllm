# vllm 自定义项目

	该项目fork自[vllm-project/vllm](https://github.com/vllm-project),  目前主要实现了对chatglm2的多卡支持
	注意：由于glm2的GQA的group_num=2, 所以目前多卡只支持2卡。

## 安装

- git clone https://github.com/David-webb/vllm.git -b vllm4chatglm
- cd vllm
- pip install -e .

## 使用

vllm/custom_launch_scripts/目录下:

- vllm_launch.sh   用于启动服务
- try_vllm.py    提供了vllm的访问接口模板（流式/非流式）
- Dockerfile   提供了构建vllm运行环境的docker镜像

## todo

- chatglm2的多卡突破GQA限制
- vllm上实现对ntk的支持
- vllm上实现flash-attention的集成（非xformer方法）
