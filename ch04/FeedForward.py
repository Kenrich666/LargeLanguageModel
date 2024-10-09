import torch
import torch.nn as nn

from ch04.GELU import GELU


class FeedForward(nn.Module):  # 定义FeedForward类，继承nn.Module
    def __init__(self, cfg):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.layers = nn.Sequential(  # 使用顺序容器定义网络层
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 线性层1
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 线性层2
        )

    def forward(self, x):  # 定义前向传播方法
        return self.layers(x)  # 返回网络层的输出
