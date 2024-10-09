import tiktoken
import torch

from ch02.GPTDatasetV1 import create_dataloader_v1
from ch04.GPTModel import GPTModel, generate_text_simple, generate
from ch05.Train import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text, train_model_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)  # 设置随机种子
file_path = "the-verdict.txt"  # 文件路径
with open(file_path, "r", encoding="utf-8") as file:  # 以读模式打开文件
    text_data = file.read()  # 读取文件内容

train_ratio = 0.90  # 训练集比例
split_idx = int(train_ratio * len(text_data))  # 计算分割索引
train_data = text_data[:split_idx]  # 获取训练数据
val_data = text_data[split_idx:]  # 获取验证数据

train_loader = create_dataloader_v1(
    train_data,  # 训练数据
    batch_size=2,  # 批大小
    max_length=GPT_CONFIG_124M["context_length"],  # 最大长度
    stride=GPT_CONFIG_124M["context_length"],  # 步幅
    drop_last=True,  # 丢弃最后一个不完整批次
    shuffle=True,  # 是否打乱数据
    num_workers=0  # 工作线程数
)

val_loader = create_dataloader_v1(
    val_data,  # 验证数据
    batch_size=2,  # 批大小
    max_length=GPT_CONFIG_124M["context_length"],  # 最大长度
    stride=GPT_CONFIG_124M["context_length"],  # 步幅
    drop_last=False,  # 不丢弃最后一个不完整批次
    shuffle=False,  # 是否打乱数据
    num_workers=0  # 工作线程数
)
model = GPTModel(GPT_CONFIG_124M)  # 初始化模型
model.to(device)  # 将模型移动到设备
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # 使用AdamW优化器
num_epochs = 10  # 训练周期数
train_losses, val_losses, tokens_seen = train_model_simple(  # 调用train_model_simple函数
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)

model.to("cpu")  # 将模型移至CPU
model.eval()  # 设置模型为评估模式

# token_ids = generate_text_simple(  # 调用generate_text_simple函数生成词元
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),  # 将文本转换为词元ID
#     max_new_tokens=25,  # 最大生成词元数为25
#     context_size=GPT_CONFIG_124M["context_length"]  # 上下文大小为GPT_CONFIG_124M的context_length
# )

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),  # 将文本转换为词元ID并移动到设备
    max_new_tokens=15,  # 最大新词元数为15
    context_size=GPT_CONFIG_124M["context_length"],  # 上下文大小
    top_k=25,  # top-k值为25
    temperature=1.4  # 温度值为1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer).replace("\n", " "))  # 打印生成的文本
