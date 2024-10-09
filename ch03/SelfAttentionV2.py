import torch
import torch.nn as nn


class SelfAttentionV2(nn.Module):  # 定义SelfAttention_v2类，继承自nn.Module
    def __init__(self, d_in, d_out, qkv_bias=False):  # 初始化方法，接受输入维度、输出维度和qkv_bias参数
        super().__init__()  # 调用父类的初始化方法
        self.d_out = d_out  # 设置输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义查询权重矩阵的线性变换
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义键权重矩阵的线性变换
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义值权重矩阵的线性变换

    def forward(self, x):  # 前向传播方法，接受输入x
        keys = self.W_key(x)  # 计算键向量
        queries = self.W_query(x)  # 计算查询向量
        values = self.W_value(x)  # 计算值向量
        attn_scores = queries @ keys.T  # 计算注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 计算注意力权重
        context_vec = attn_weights @ values  # 计算上下文向量
        return context_vec  # 返回上下文向量


if __name__ == '__main__':
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
    print(sa_v2(inputs))  # 打印实例的上下文向量
