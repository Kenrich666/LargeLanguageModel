import torch

from ch04.GPTModel import GPTModel

GPT_CONFIG_124M = {  # GPT配置字典
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # A 将上下文长度从1024缩短到256词元
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # B 可能且常见的是将dropout设置为0。
    "qkv_bias": False  # QKV偏置
}
model = GPTModel(GPT_CONFIG_124M)  # 初始化新模型实例
# model.load_state_dict(torch.load("model.pth"))  # 加载保存的模型权重
checkpoint = torch.load("model_and_optimizer.pth")  # 加载保存的检查点
model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型状态字典
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)  # 初始化优化器
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器状态字典
model.train()  # 设置模型为训练模式


model.eval()  # 设置模型为评估模式
