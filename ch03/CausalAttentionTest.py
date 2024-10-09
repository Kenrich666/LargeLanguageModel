import torch  # 导入torch库

from ch03.SelfAttentionV2 import SelfAttentionV2

inputs = torch.tensor(
 [[0.43, 0.15, 0.89],  # "Your" 对应的嵌入向量 (x^1)
  [0.55, 0.87, 0.66],  # "journey" 对应的嵌入向量 (x^2)
  [0.57, 0.85, 0.64],  # "starts" 对应的嵌入向量 (x^3)
  [0.22, 0.58, 0.33],  # "with" 对应的嵌入向量 (x^4)
  [0.77, 0.25, 0.10],  # "one" 对应的嵌入向量 (x^5)
  [0.05, 0.80, 0.55]]  # "step" 对应的嵌入向量 (x^6)
)

x_2 = inputs[1]  # A # 第二个输入元素
d_in = inputs.shape[1]  # B # 输入嵌入大小，d=3
d_out = 2  # C # 输出嵌入大小，d_out=2

torch.manual_seed(123)  # 为了可重复的随机初始化，设置随机种子
sa_v2 = SelfAttentionV2(d_in, d_out)  # 创建SelfAttention_v1实例

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

context_length = attn_scores.shape[0]  # 获取上下文长度
# mask_simple = torch.tril(torch.ones(context_length, context_length))  # 创建简单掩码
# print(mask_simple)  # 打印简单掩码

# masked_simple = attn_weights * mask_simple  # 将掩码与注意力权重相乘
# print(masked_simple)  # 打印掩码后的注意力权重
#
# row_sums = masked_simple.sum(dim=1, keepdim=True)  # 计算每行的总和
# masked_simple_norm = masked_simple / row_sums  # 将每行的元素除以每行的总和
# print(masked_simple_norm)  # 打印重新规范化后的注意力权重

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 创建掩码
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)  # 将掩码位置填充为负无穷大
# print(masked)  # 打印掩码后的得分

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

torch.manual_seed(123)  # 设置随机种子以确保结果可重复
dropout = torch.nn.Dropout(0.5)  # 创建一个dropout层，dropout率为50%
print(dropout(attn_weights))  # 打印应用dropout后的注意力权重矩阵
