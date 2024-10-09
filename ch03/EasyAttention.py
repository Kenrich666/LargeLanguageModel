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

query = inputs[1]  # 将第二个输入词元 ("journey") 作为查询向量
attn_scores_2 = torch.empty(inputs.shape[0])  # 初始化一个张量，用于存储每个词元的注意力得分
for i, x_i in enumerate(inputs):  # 遍历每个输入词元
    attn_scores_2[i] = torch.dot(x_i, query)  # 计算查询向量与当前输入词元之间的点积（注意力得分）
# print(attn_scores_2)  # 打印计算出的注意力得分

# 计算注意力得分的归一化权重
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()


# 打印注意力权重，结果应为归一化后的注意力得分
# print("Attention weights:", attn_weights_2_tmp)
# 输出: Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])

# 打印注意力权重的总和，验证归一化是否正确
# print("Sum:", attn_weights_2_tmp.sum())


# 输出: Sum: tensor(1.0000)

# 定义一个简单的softmax函数，用于计算注意力权重
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)  # 计算输入x的指数并进行归一化


# 使用简单的softmax函数计算注意力权重
attn_weights_2 = softmax_naive(attn_scores_2)

# 打印注意力权重
# print("Attention weights:", attn_weights_2_naive)
# 输出: Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

# 打印注意力权重的总和，验证softmax归一化是否正确
# print("Sum:", attn_weights_2_naive.sum())
# 输出: Sum: tensor(1.)

query = inputs[1]  # 将第二个输入词元 ("journey") 作为查询向量
context_vec_2 = torch.zeros(query.shape)  # 初始化上下文向量为零向量，形状与查询向量相同

# 遍历每个输入词元，计算其与注意力权重的乘积，并累加到上下文向量中
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# 打印计算出的上下文向量
# print(context_vec_2)
# 输出: tensor([0.4419, 0.6515, 0.5683])

# attn_scores = torch.empty(6, 6)  # 创建一个6x6的张量用于存储注意力得分
# for i, x_i in enumerate(inputs):  # 遍历每个输入词元
#     for j, x_j in enumerate(inputs):  # 再次遍历每个输入词元
#         attn_scores[i, j] = torch.dot(x_i, x_j)  # 计算两个输入词元的点积
# print(attn_scores)  # 打印计算出的注意力得分

attn_scores = inputs @ inputs.T  # 通过矩阵乘法计算注意力得分
# print(attn_scores)  # 打印注意力得分

attn_weights = torch.softmax(attn_scores, dim=-1)  # 对每一行进行softmax归一化
# print(attn_weights)  # 打印归一化后的注意力权重

all_context_vecs = attn_weights @ inputs  # 通过矩阵乘法计算所有上下文向量
# print(all_context_vecs)  # 打印所有上下文向量

print("Previous 2nd context vector:", context_vec_2)  # 打印之前计算的第二个上下文向量

