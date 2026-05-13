# 特性全开时乱码
* 输出情况：
 * 打开DSA，打开SWA时，输出正常
 * 打开DSA，关闭SWA时，输出乱码
 * 关闭DSA，打开SWA时，输出乱码
 * 关闭DSA，关闭SWA时，输出说人话但是上下文明显不对

* backend选择情况：
 * 打开DSA，打开SWA时，打印了两条log
  * Using FLASHMLA_SPARSE attention backend out of potential backends: ('FLASHMLA_SPARSE',)
  * Using FLASH_ATTN_MLA attention backend out of potential backends: ('FLASH_ATTN_MLA', 'FLASHMLA', 'TRITON_MLA')
 * 打开DSA，关闭SWA时，打印了两条log
  * Using FLASHMLA_SPARSE attention backend out of potential backends: ('FLASHMLA_SPARSE',)
  * Using FLASH_ATTN_MLA attention backend out of potential backends: ('FLASH_ATTN_MLA', 'FLASHMLA', 'TRITON_MLA')
 * 关闭DSA，打开SWA时，打印一条log
  * Using FLASHMLA_SPARSE attention backend out of potential backends: ('FLASHMLA_SPARSE',)
 * 关闭DSA，关闭SWA时，打印一条log
  * Using FLASHMLA_SPARSE attention backend out of potential backends: ('FLASHMLA_SPARSE',)


# 入图精度异常
* 不入图打印：
(Worker_TP1 pid=903586) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([4, 1024]), hidden_states.sum(): -2.21875
(Worker_TP0 pid=903585) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([4, 1024]), hidden_states.sum(): -2.21875
(Worker_TP1 pid=903586) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([1, 1024]), hidden_states.sum(): 25.25
(Worker_TP0 pid=903585) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([1, 1024]), hidden_states.sum(): 25.25
(Worker_TP1 pid=903586) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([1, 1024]), hidden_states.sum(): 5.5
(Worker_TP0 pid=903585) [DEBUG] first layer MoME forward, state_indice: 0, hidden_states.shape: torch.Size([1, 1024]), hidden_states.sum(): 5.5

* PIECEWISE入图打印：