import tiktoken
import torch

from ch04.GPTModel import GPTModel, generate
from ch05.Train import text_to_token_ids, token_ids_to_text

GPT_CONFIG = {  # GPT配置字典
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # A 将上下文长度从1024缩短到256词元
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # B 可能且常见的是将dropout设置为0。
    "qkv_bias": False  # QKV偏置
}
tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的分词器编码

model_configs = {  # 定义模型配置字典
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-xl (1558M)"  # 选择模型名称
NEW_CONFIG = GPT_CONFIG.copy()  # 复制原始配置
NEW_CONFIG.update(model_configs[model_name])  # 更新配置为选定模型的配置
NEW_CONFIG.update({"context_length": 1024})  # 更新上下文长度为1024
NEW_CONFIG.update({"qkv_bias": True})  # 启用偏置向量

if __name__ == '__main__':
    gpt = GPTModel(NEW_CONFIG)  # 使用更新的配置初始化GPT模型
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)  # 使用AdamW优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("model_and_optimizer.pth")  # 加载保存的检查点
    gpt.load_state_dict(checkpoint["model_state_dict"])  # 加载模型状态字典
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)  # 初始化优化器
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器状态字典

    gpt.to(device)  # 将模型移动到设备

    gpt.eval()  # 设置模型为评估模式

    torch.manual_seed(123)  # 设置随机种子
    token_ids = generate(  # 生成新文本
        model=gpt,
        idx=text_to_token_ids("I am", tokenizer).to(device),
        max_new_tokens=20,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印输出文本
