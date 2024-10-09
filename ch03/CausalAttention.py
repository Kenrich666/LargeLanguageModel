import torch
import torch.nn as nn


class CausalAttention(nn.Module):  # 定义CausalAttention类，继承自nn.Module
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.d_out = d_out  # 设置输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义查询权重矩阵
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义键权重矩阵
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义值权重矩阵
        self.dropout = nn.Dropout(dropout)  # 定义dropout层
        self.register_buffer(  # 注册缓冲区，用于存储因果掩码
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 创建上三角掩码
        )

    def forward(self, x):  # 前向传播方法
        b, num_tokens, d_in = x.shape  # 获取批次大小、令牌数量和输入维度
        keys = self.W_key(x)  # 计算键向量
        queries = self.W_query(x)  # 计算查询向量
        values = self.W_value(x)  # 计算值向量
        attn_scores = queries @ keys.transpose(1, 2)  # 计算查询与键的点积作为注意力得分
        attn_scores.masked_fill_(  # 应用掩码，将掩码位置填充为负无穷大
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 使用softmax函数对注意力得分进行归一化
        attn_weights = self.dropout(attn_weights)  # 应用dropout
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
    d_in = inputs.shape[1]  # B # 输入嵌入大小，d=3
    d_out = 2  # C # 输出嵌入大小，d_out=2

    batch = torch.stack((inputs, inputs), dim=0)  # 将输入张量堆叠，模拟批次输入
    # print(batch.shape)  # 打印批次输入的形状

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)