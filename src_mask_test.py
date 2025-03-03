import torch
import numpy as np

def create_padding_mask(seq, pad_token=0):
    """
    创建 Padding Mask，填充值为0的地方 mask 为 1，其他位置为 0
    这样在注意力计算时，被 Mask 住的地方（填充部分）将被赋值 -inf，最终 Softmax 变为 0
    """
    return (seq == pad_token).unsqueeze(1)  # [batch_size, 1, 1, seq_len]

# 示例输入序列（batch_size=2, seq_len=5）
input_seq = torch.tensor([
    [1, 2, 3, 4, 0],
      [2, 3, 4, 0, 0]  # 0 是填充值
])

# 生成 Padding Mask
padding_mask = create_padding_mask(input_seq)
# print(padding_mask.shape)


mask = padding_mask.expand(2, 5, 5)
print(mask.shape)
# 