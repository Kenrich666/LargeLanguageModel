import torch  # 导入torch库
import torch.nn as nn  # 导入torch.nn库并重命名为nn


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承nn.Module
    def __init__(self, emb_dim):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.eps = 1e-5  # 设置epsilon值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 定义可训练的scale参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 定义可训练的shift参数

    def forward(self, x):  # 定义前向传播方法
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 进行归一化
        return self.scale * norm_x + self.shift  # 返回归一化后的输出
