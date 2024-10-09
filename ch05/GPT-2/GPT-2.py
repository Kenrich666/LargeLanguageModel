import tiktoken
import torch
import numpy as np  # 导入numpy库

from ch04.GPTModel import GPTModel, generate
from ch05.Train import text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2  # 从gpt_download导入download_and_load_gpt2函数

GPT_CONFIG = {  # GPT配置字典
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # A 将上下文长度从1024缩短到256词元
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # B 可能且常见的是将dropout设置为0。
    "qkv_bias": False  # QKV偏置
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def assign(left, right):  # 定义assign函数
    if left.shape != right.shape:  # 如果形状不匹配
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")  # 抛出错误
    return torch.nn.Parameter(torch.tensor(right))  # 返回右张量作为可训练参数


def load_weights_into_gpt(gpt, params):  # 定义load_weights_into_gpt函数
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])  # A 分配位置嵌入权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])  # A 分配词元嵌入权重

    for b in range(len(params["blocks"])):  # B 遍历块
        q_w, k_w, v_w = np.split(  # C 分割权重
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=0)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)  # 分配MLP层权重
        gpt.trf_blocks[b].ff.layers[0].bias = assign(  # 分配MLP层偏置
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(  # 分配MLP层权重
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(  # 分配MLP层偏置
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(  # 分配第一个LayerNorm层权重
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(  # 分配第一个LayerNorm层偏置
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(  # 分配第二个LayerNorm层权重
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(  # 分配第二个LayerNorm层偏置
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  # 分配最终LayerNorm层权重
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])  # 分配最终LayerNorm层偏置
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # D 将输出层权重与词元嵌入层权重绑定


settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")  # 下载并加载GPT-2设置和参数
# print("Settings:", settings)  # 打印设置
# print("Parameter dictionary keys:", params.keys())  # 打印参数字典键

# print(params["wte"])  # 打印词元嵌入权重张量
# print("Token embedding weight tensor dimensions:", params["wte"].shape)  # 打印词元嵌入权重张量的维度

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

tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的分词器编码

gpt = GPTModel(NEW_CONFIG)  # 使用更新的配置初始化GPT模型
optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)  # 使用AdamW优化器

load_weights_into_gpt(gpt, params)  # 加载权重到GPT模型

torch.save({  # 保存模型和优化器的状态字典
    "model_state_dict": gpt.state_dict(),  # 模型状态字典
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态字典
}, "model_and_optimizer.pth")  # 保存到model_and_optimizer.pth文件
gpt.to(device)  # 将模型移动到设备

gpt.eval()  # 设置模型为评估模式

torch.manual_seed(123)
token_ids = generate(  # 生成新文本
    model=gpt,
    idx=text_to_token_ids("I am", tokenizer).to(device),
    max_new_tokens=100,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印输出文本
