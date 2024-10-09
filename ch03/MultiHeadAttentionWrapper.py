import torch
import torch.nn as nn

from ch03.CausalAttention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):  # 定义一个MultiHeadAttentionWrapper类，继承自nn.Module
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):  # 初始化方法，定义了输入、输出、上下文长度、dropout率、头数量和是否使用qkv_bias
        super().__init__()  # 调用父类的初始化方法
        self.heads = nn.ModuleList(  # 创建一个ModuleList存储多个CausalAttention模块的实例
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]  # 根据头数量创建多个CausalAttention实例
        )

    def forward(self, x):  # 前向传播方法
        return torch.cat([head(x) for head in self.heads], dim=-1)  # 对每个头的输出进行拼接，并沿最后一个维度连接

if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # "Your" 对应的嵌入向量 (x^1)
         [0.55, 0.87, 0.66],  # "journey" 对应的嵌入向量 (x^2)
         [0.57, 0.85, 0.64],  # "starts" 对应的嵌入向量 (x^3)
         [0.22, 0.58, 0.33],  # "with" 对应的嵌入向量 (x^4)
         [0.77, 0.25, 0.10],  # "one" 对应的嵌入向量 (x^5)
         [0.05, 0.80, 0.55]]  # "step" 对应的嵌入向量 (x^6)
    )

    batch = torch.stack((inputs, inputs), dim=0)  # 将输入张量堆叠，模拟批次输入

    torch.manual_seed(123)  # 设定随机种子
    context_length = batch.shape[1]  # 获取词元的数量
    d_in, d_out = 3, 2  # 定义输入和输出的维度
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)  # 初始化多头注意力包装器
    context_vecs = mha(batch)  # 计算上下文向量
    print(context_vecs)  # 打印上下文向量
    print("context_vecs.shape:", context_vecs.shape)  # 打印上下文向量的形状
