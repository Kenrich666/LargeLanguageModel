import torch  # 导入torch库

# 定义一个名为inputs的张量，包含6个3维向量，每个向量表示一个嵌入词元
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
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 初始化查询权重矩阵W_q
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 初始化键权重矩阵W_k
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 初始化值权重矩阵W_v

query_2 = x_2 @ W_query  # 计算查询向量q^(2)
key_2 = x_2 @ W_key  # 计算键向量k^(2)
value_2 = x_2 @ W_value  # 计算值向量v^(2)
# print(query_2)  # 打印查询向量

keys = inputs @ W_key
values = inputs @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)

keys_2 = keys[1]  # A # 记住Python从0开始索引
attn_score_22 = query_2.dot(keys_2)  # 计算查询向量q^(2)和键向量k^(2)的点积
# print(attn_score_22)  # 打印注意力分数

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
# print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values  # 通过注意力权重计算上下文向量
print(context_vec_2)  # 打印上下文向量
