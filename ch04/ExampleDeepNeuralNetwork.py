import torch
import torch.nn as nn

from ch04.GELU import GELU


class ExampleDeepNeuralNetwork(nn.Module):  # 定义ExampleDeepNeuralNetwork类，继承nn.Module
    def __init__(self, layer_sizes, use_shortcut):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.use_shortcut = use_shortcut  # 设置是否使用shortcut
        self.layers = nn.ModuleList([  # 使用模块列表定义网络层
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),  # 实现5层
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):  # 定义前向传播方法
        for layer in self.layers:  # 遍历每一层
            layer_output = layer(x)  # 计算当前层的输出
            if self.use_shortcut and x.shape == layer_output.shape:  # 检查是否可以应用shortcut
                x = x + layer_output  # 应用shortcut
            else:
                x = layer_output  # 否则直接输出当前层结果
        return x  # 返回最终输出


def print_gradients(model, x):  # 定义打印梯度的函数
    # Forward pass
    output = model(x)  # 前向传播计算输出
    target = torch.tensor([[0.]])  # 目标张量

    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()  # 使用均方误差损失
    loss = loss(output, target)  # 计算损失

    # Backward pass to calculate the gradients
    loss.backward()  # 反向传播计算梯度

    for name, param in model.named_parameters():  # 遍历模型的每个参数
        if 'weight' in name:  # 如果参数名中包含'weight'
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")  # 打印权重的平均绝对梯度


if __name__ == '__main__':
    layer_sizes = [3, 3, 3, 3, 3, 1]  # 定义每层的大小
    sample_input = torch.tensor([[1., 0., -1.]])  # 样本输入
    torch.manual_seed(123)  # 指定初始权重的随机种子以确保结果可复现
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)  # 初始化没有shortcut的神经网络
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)  # 初始化有shortcut的神经网络
    print("without shortcut:")
    print_gradients(model_without_shortcut, sample_input)  # 打印没有shortcut的模型的梯度
    print("with shortcut:")
    print_gradients(model_with_shortcut, sample_input)  # 打印没有shortcut的模型的梯度
