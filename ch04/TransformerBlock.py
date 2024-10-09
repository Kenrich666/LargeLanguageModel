import torch
import torch.nn as nn
from ch03.MultiHeadAttention import MultiHeadAttention
from ch04.Config import GPT_CONFIG_124M
from ch04.FeedForward import FeedForward
from ch04.LayerNorm import LayerNorm


class TransformerBlock(nn.Module):  # 定义TransformerBlock类，继承nn.Module
    def __init__(self, cfg):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.att = MultiHeadAttention(  # 实例化多头注意力
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)  # 实例化前馈层
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 实例化层归一化1
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 实例化层归一化2
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 实例化dropout

    def forward(self, x):  # 定义前向传播方法
        shortcut = x  # 保存输入以便添加shortcut
        x = self.norm1(x)  # 层归一化1
        x = self.att(x)  # 多头注意力
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut  # 添加原始输入回去

        shortcut = x  # 保存输入以便添加shortcut
        x = self.norm2(x)  # 层归一化2
        x = self.ff(x)  # 前馈层
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut  # 添加原始输入回去

        return x  # 返回最终输出


if __name__ == '__main__':
    torch.manual_seed(123)  # 设置随机数种子为123，以确保生成的随机数具有可重复性
    x = torch.rand(2, 4, 768)  # 创建一个形状为[2, 4, 768]的随机张量，表示输入数据，包含2个批次，每个批次4个token，每个token用768维向量表示
    block = TransformerBlock(GPT_CONFIG_124M)  # 实例化一个TransformerBlock类，使用配置字典GPT_CONFIG_124M
    output = block(x)  # 将输入张量x传递给TransformerBlock实例，获取输出

    print("Input shape:", x.shape)  # 打印输入张量的形状
    print("Output shape:", output.shape)  # 打印输出张量的形状
