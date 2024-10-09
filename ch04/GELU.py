import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块并重命名为plt


class GELU(nn.Module):  # 定义GELU类，继承nn.Module
    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

    def forward(self, x):  # 定义前向传播方法
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    gelu, relu = GELU(), nn.ReLU()  # 创建GELU和ReLU实例

    x = torch.linspace(-3, 3, 100)  # 在-3到3的范围内创建100个样本数据点
    y_gelu, y_relu = gelu(x), relu(x)  # 分别计算GELU和ReLU的输出
    plt.figure(figsize=(8, 3))  # 设置图形大小
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):  # 枚举GELU和ReLU输出及其标签
        plt.subplot(1, 2, i)  # 创建子图
        plt.plot(x, y)  # 绘制函数图形
        plt.title(f"{label} activation function")  # 设置图形标题
        plt.xlabel("x")  # 设置x轴标签
        plt.ylabel(f"{label}(x)")  # 设置y轴标签
        plt.grid(True)  # 显示网格线
    plt.tight_layout()  # 紧凑布局
    plt.show()  # 显示图形
