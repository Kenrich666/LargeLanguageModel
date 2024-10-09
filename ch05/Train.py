import tiktoken
import torch
import matplotlib
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库作为plt

from ch02.GPTDatasetV1 import create_dataloader_v1
from ch04.GPTModel import generate_text_simple, GPTModel

GPT_CONFIG_124M = {  # GPT配置字典
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # A 将上下文长度从1024缩短到256词元
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # B 可能且常见的是将dropout设置为0。
    "qkv_bias": False  # QKV偏置
}


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):  # 定义plot_losses函数
    fig, ax1 = plt.subplots(figsize=(5, 3))  # 创建一个图形和一个子图，图形大小为5x3
    ax1.plot(epochs_seen, train_losses, label="Training loss")  # 在第一个子图上绘制训练损失
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")  # 在第一个子图上绘制验证损失，使用点线样式
    ax1.set_xlabel("Epochs")  # 设置x轴标签为“Epochs”
    ax1.set_ylabel("Loss")  # 设置y轴标签为“Loss”
    ax1.legend(loc="upper right")  # 设置图例位置为右上角
    ax2 = ax1.twinx()  # 创建共享同一y轴的第二个x轴  #A
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 对齐刻度的隐形图  #B
    ax2.set_xlabel("Tokens seen")  # 设置第二个x轴标签为“Tokens seen”
    fig.tight_layout()  # 自动调整子图参数以填充整个图形区域
    plt.show()  # 显示图形



def text_to_token_ids(text, tokenizer):  # 定义text_to_token_ids函数
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  # 编码文本，允许特殊词元
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor  # 返回编码后的张量


def token_ids_to_text(token_ids, tokenizer):  # 定义token_ids_to_text函数
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 解码为文本


def calc_loss_batch(input_batch, target_batch, model, device):  # 定义计算批次损失的函数
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标批次转移到设备上
    logits = model(input_batch)  # 模型计算logits
    loss = torch.nn.functional.cross_entropy(  # 计算交叉熵损失
        logits.flatten(0, 1), target_batch.flatten()  # 展平logits和目标批次
    )
    return loss  # 返回损失


def calc_loss_loader(data_loader, model, device, num_batches=None):  # 定义计算加载器损失的函数
    total_loss = 0.  # 初始化总损失为0
    if len(data_loader) == 0:  # 如果加载器为空
        return float("nan")  # 返回NaN
    elif num_batches is None:  # 如果未指定批次数
        num_batches = len(data_loader)  # 使用加载器中的批次数
    else:
        num_batches = min(num_batches, len(data_loader))  # 限制批次数为加载器中的批次数
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历加载器中的批次
        if i < num_batches:  # 如果未达到指定批次数
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算批次损失
            total_loss += loss.item()  # 累加损失
        else:
            break  # 超过指定批次数则退出
    return total_loss / num_batches  # 返回平均损失


def evaluate_model(model, train_loader, val_loader, device, eval_iter):  # 定义evaluate_model函数
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度跟踪
        train_loss = calc_loss_loader(train_loader, model, device,  # 计算训练集损失
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device,  # 计算验证集损失
                                    num_batches=eval_iter)
    model.train()  # 设置模型为训练模式
    return train_loss, val_loss  # 返回训练和验证损失


def generate_and_print_sample(model, tokenizer, device, start_context):  # 定义generate_and_print_sample函数
    model.eval()  # 设置模型为评估模式
    context_size = model.pos_emb.weight.shape[0]  # 获取上下文大小
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # 将文本转换为词元ID并移动到设备
    with torch.no_grad():  # 禁用梯度跟踪
        token_ids = generate_text_simple(  # 生成文本词元ID
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)  # 将词元ID转换为文本
    print(decoded_text.replace("\n", " "))  # 打印生成的文本，以紧凑格式显示
    model.train()  # 设置模型为训练模式


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,  # 定义train_model_simple函数
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []  # 初始化列表以跟踪损失和看到的词元
    tokens_seen, global_step = 0, -1  # 初始化词元计数和全局步数

    for epoch in range(num_epochs):  # 开始主要训练循环
        model.train()  # 设置模型为训练模式
        for input_batch, target_batch in train_loader:  # 遍历训练数据
            optimizer.zero_grad()  # 重置前一批次迭代的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算批次损失
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            tokens_seen += input_batch.numel()  # 更新词元计数
            global_step += 1  # 增加全局步数

            if global_step % eval_freq == 0:  # 可选的评估步骤
                train_loss, val_loss = evaluate_model(  # 评估模型性能
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # 添加训练损失到列表
                val_losses.append(val_loss)  # 添加验证损失到列表
                track_tokens_seen.append(tokens_seen)  # 记录看到的词元数
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "  # 打印当前训练信息
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(  # 生成并打印样本
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen  # 返回训练和验证损失及词元计数


if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的分词器编码
    device = "cuda"
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
    # torch.save(model.state_dict(), "model.pth")  # 保存模型权重到model.pth文件
    torch.save({  # 保存模型和优化器的状态字典
        "model_state_dict": model.state_dict(),  # 模型状态字典
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态字典
    }, "model_and_optimizer.pth")  # 保存到model_and_optimizer.pth文件
