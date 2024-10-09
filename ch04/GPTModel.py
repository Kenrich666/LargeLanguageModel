import torch
import torch.nn as nn
import tiktoken

from ch04.Config import GPT_CONFIG_124M
from ch04.TransformerBlock import TransformerBlock
from ch04.LayerNorm import LayerNorm


class GPTModel(nn.Module):  # 定义GPTModel类，继承nn.Module
    def __init__(self, cfg):  # 初始化方法，接受配置参数cfg
        super().__init__()  # 调用父类的初始化方法
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 创建分词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 创建位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 创建dropout层
        self.trf_blocks = nn.Sequential(  # 创建一系列Transformer块
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])  # 创建最终层归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 创建输出层

    def forward(self, in_idx):  # 前向传播方法
        batch_size, seq_len = in_idx.shape  # 获取输入的批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取分词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将分词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用dropout
        x = self.trf_blocks(x)  # 通过Transformer块
        x = self.final_norm(x)  # 应用最终层归一化
        logits = self.out_head(x)  # 通过输出层得到logits
        return logits  # 返回logits


def generate_text_simple(model, idx, max_new_tokens, context_size):  # idx是当前上下文中的（batch，n_tokens）索引数组
    for _ in range(max_new_tokens):  # 循环生成新词元
        idx_cond = idx[:, -context_size:]  # 如果当前上下文超过支持的上下文大小，则裁剪当前上下文。例如，如果LLM仅支持5个令牌，而上下文大小为10，则仅使用最后5个令牌作为上下文
        with torch.no_grad():  # 禁用梯度计算
            logits = model(idx_cond)  # 通过模型获得logits

        logits = logits[:, -1, :]  # 只关注最后一个时间步的logits
        probas = torch.softmax(logits, dim=-1)  # 应用softmax函数获取概率分布
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # 选择具有最高概率的词元ID
        idx = torch.cat([idx, idx_next], dim=1)  # 将生成的词元ID附加到当前上下文
    return idx  # 返回生成的词元序列


if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2编码器
    start_context = "Hello, I am"  # 定义初始上下文
    encoded = tokenizer.encode(start_context)  # 使用分词器编码初始上下文
    # print("encoded:", encoded)  # 打印编码后的词元ID
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将编码后的词元ID转换为张量并添加批次维度
    # print("encoded_tensor.shape:", encoded_tensor.shape)  # 打印张量的形状

    model = GPTModel(GPT_CONFIG_124M)

    model.eval()  # 将模型置于评估模式
    out = generate_text_simple(  # 使用generate_text_simple函数生成文本
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)  # 打印输出张量
    print("Output length:", len(out[0]))  # 打印输出张量的长度

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())  # 解码输出张量为文本
    print(decoded_text)  # 打印解码后的文本
