# MultiStream介绍

本文档介绍了Maya-VLLM multi-stream特性的使用及模型改写方案。

## 特性介绍

Multi-stream针对大数据量下的计算和通信互相阻塞问题，将模型执行过程拆解成多个流进行计算和通信的并发，
达到减少模型推理时延的目的，具体方案介绍可参见[ATA FlexEngine通算并发](https://ata.atatech.org/articles/12020343639)。

## 使用方式
启动服务时，添加如下参数即可：
```shell
--enable-multi-stream
```

## 模型使能MS修改说明
经典transformer架构模型，且使用Causal Mask的模型都可通过少量代码改造，完成MS特性支持。
以antglm模型为例，总共包含下列4个步骤：
### Step 1. 添加MS相关引用
```python
from vllm.config import MultiStreamConfig
from .interfaces import SupportsMultiStream
from vllm.multistream.layers import (MultiStreamPreTransformerLayer, MultiStreamPostTransformerLayer)
from vllm.multistream.metadata import make_multistream_metadata
from vllm.multistream.decorators import support_multi_stream
```

### Step 2. 修改CausalLM类，声明支持MS
```python
class GLMForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsMultiStream):  # 添加SupportsMultiStream:
    def __init__(self,
                 config: GLMConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None,
                 multistream_config: Optional[MultiStreamConfig] = None, # 添加MS config
                 ):
        ...
        
        self.transformer = GLMModel(
            config,
            cache_config,
            quant_config,
            multistream_config=multistream_config,  # 构建Model时传入MS Config
            prefix="transformer",
        )
        ...
```

### Step 3. 修改Model类
```python
class GLMModel(nn.Module):
    def __init__(
            self,
            config: GLMConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            multistream_config: Optional[MultiStreamConfig] = None,  # 添加MS Config
            prefix: str = "",
    ):
        ...
        # 构建MS metadata和layer
        self.multistream_metadata = make_multistream_metadata(
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            causal_lm=getattr(config, "causal_lm", True),  # 是否是causal_lm
            multistream_config=multistream_config,
        )
        # 使能MS的前置切分layer
        self.ms_pre_layer = MultiStreamPreTransformerLayer(self.multistream_metadata)
        # 使能MS的后置汇总layer
        self.ms_post_layer = MultiStreamPostTransformerLayer(self.multistream_metadata)

    def forward(self, ...):
        ...  # 处理pp等输入等逻辑
        # 对attn_metadata等推理输入数据进行切分，可自动判断是否能够切分
        attn_metadata, [position_ids, hidden_states, residual] = self.ms_pre_layer(
            attn_metadata,
            [position_ids, hidden_states, residual],
        )
        # 循环执行TransformerBlock，无需更改
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                ...
            )
        # 对MS切分或者不切分的数据进行聚合
        [hidden_states, residual] = self.ms_post_layer(
            [hidden_states, residual],
        )
        # 后续处理按原始版本
        ...
```

### Step4. 修改TransformerLayer
#### Step 4.1. 将Transformer Layer的`forward`拆分为`forward_attn`和`forward_ffn`部分
注意：
1. `forward_attn`输入为原始`forward`输入，`forward_ffn`输入为`forward_attn`输出
2. `forward_attn`处理attention部分，结尾至attention算子，`forward_ffn`处理ffn部分，结尾为mlp算子
#### Step 4.2. 用support_multi_stream装饰器装饰TransformerLayer
关键字释义：
1. `dynamic_arg_ms`，标识`forward`、`forward_attn`、`forward_ffn`三个函数的输入参数名，顺序与实际函数相同
2. `unpacked_arg`，标识上述函数输出参数中，不可通过ms参数进行切分的参数，典型如`kv_cache`（vllm的kv cache是所有kv的存储tensor），
如果有其它表示性参数，如各种flag，则将其加入到`unpacked_arg`中。

```python
@support_multi_stream(
    dynamic_arg_ms={
        "forward": ["hidden_states", "position_ids", "kv_cache", "attn_metadata", "residual"],
        "forward_attn": ["hidden_states", "position_ids", "kv_cache", "attn_metadata", "residual"],
        "forward_ffn": ["hidden_states", "residual"],
    },
    unpacked_arg={"kv_cache",},
)
class GLMBlock(nn.Module):
    ...
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCache,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
    ):
        ...
    
    # 添加forward_attn函数，处理attn部分，该部分结束在attention算子
    def forward_attn(self,
                     hidden_states: torch.Tensor,
                     position_ids: torch.Tensor,
                     kv_cache: KVCache,
                     attn_metadata: AttentionMetadata,
                     residual: Optional[torch.Tensor],
                     ):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        return hidden_states, residual

    # 添加forward_ffn函数，处理ffn部分，该部分结束在mlp算子
    def forward_ffn(self,
                    hidden_states: torch.Tensor,
                    residual: torch.Tensor,):
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```


