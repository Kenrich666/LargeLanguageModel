import torch.nn as nn  # 导入torch.nn库并重命名为nn


class DummyGPTModel(nn.Module):  # 定义DummyGPTModel类，继承nn.Module
    def __init__(self, cfg):  # 定义初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 定义词元嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 定义位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 定义丢弃层
        self.trf_blocks = nn.Sequential(  # 定义顺序容器，用于存放transformer块
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 使用占位transformer块
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])  # 使用占位层归一化
        self.out_head = nn.Linear(  # 定义线性层
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):  # 定义前向传播方法
        batch_size, seq_len = in_idx.shape  # 获取批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取词元嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 合并嵌入
        x = self.drop_emb(x)  # 应用丢弃层
        x = self.trf_blocks(x)  # 应用transformer块
        x = self.final_norm(x)  # 应用归一化层
        logits = self.out_head(x)  # 计算输出
        return logits  # 返回输出


class DummyTransformerBlock(nn.Module):  # 定义占位transformer块类，继承nn.Module
    def __init__(self, cfg):  # 定义初始化方法
        super().__init__()  # 调用父类的初始化方法

    def forward(self, x):  # 定义前向传播方法
        return x  # 返回输入


class DummyLayerNorm(nn.Module):  # 定义占位层归一化类，继承nn.Module
    def __init__(self, normalized_shape, eps=1e-5):  # 定义初始化方法
        super().__init__()  # 调用父类的初始化方法

    def forward(self, x):  # 定义前向传播方法
        return x  # 返回输入


if __name__ == '__main__':
    import tiktoken  # 导入tiktoken库
    import torch  # 导入torch库
    import torch.nn as nn  # 导入torch.nn库并重命名为nn

    from ch04.Config import GPT_CONFIG_124M

    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2编码器
    batch = []  # 初始化批处理列表
    txt1 = "Every effort moves you"  # 文本1
    txt2 = "Every day holds a"  # 文本2

    batch.append(torch.tensor(tokenizer.encode(txt1)))  # 将文本1编码并添加到批处理列表
    batch.append(torch.tensor(tokenizer.encode(txt2)))  # 将文本2编码并添加到批处理列表
    batch = torch.stack(batch, dim=0)  # 将批处理列表堆叠成张量
    # print(batch)  # 输出批处理张量

    torch.manual_seed(123)  # 设置随机种子
    model = DummyGPTModel(GPT_CONFIG_124M)  # 创建DummyGPTModel实例
    logits = model(batch)  # 获取模型输出logits
    print("Output shape:", logits.shape)  # 输出logits的形状
    print(logits)  # 输出logits