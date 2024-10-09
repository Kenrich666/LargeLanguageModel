import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):  # 定义多头注意力类，继承自 nn.Module
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):  # 初始化方法，定义输入、输出、上下文长度、dropout率、头的数量和是否使用偏置
        super().__init__()  # 调用父类的初始化方法
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保输出维度能被头的数量整除
        self.d_out = d_out  # 保存输出维度
        self.num_heads = num_heads  # 保存头的数量
        self.head_dim = d_out // num_heads  # 计算每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义查询权重矩阵
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义键权重矩阵
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 定义值权重矩阵
        self.out_proj = nn.Linear(d_out, d_out)  # 定义输出投影矩阵
        self.dropout = nn.Dropout(dropout)  # 定义 dropout 层
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # 注册一个上三角矩阵作为掩码，用于防止信息泄露

    def forward(self, x):  # 前向传播方法
        b, num_tokens, d_in = x.shape  # 获取批量大小、词元数量和输入维度
        keys = self.W_key(x)  # 计算键
        queries = self.W_query(x)  # 计算查询
        values = self.W_value(x)  # 计算值

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # 将键张量 reshape
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # 将值张量 reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # 将查询张量 reshape

        keys = keys.transpose(1, 2)  # 转置键张量
        queries = queries.transpose(1, 2)  # 转置查询张量
        values = values.transpose(1, 2)  # 转置值张量

        attn_scores = queries @ keys.transpose(2, 3)  # 计算注意力分数
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 获取掩码
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 应用掩码

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 计算注意力权重
        attn_weights = self.dropout(attn_weights)  # 应用 dropout

        context_vec = (attn_weights @ values).transpose(1, 2)  # 计算上下文向量
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # 调整上下文向量的形状
        context_vec = self.out_proj(context_vec)  # 应用输出投影
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

    batch = torch.stack((inputs, inputs), dim=0)  # 将输入张量堆叠，模拟批次输入

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
