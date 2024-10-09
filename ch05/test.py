from ch04.GPTModel import GPTModel, generate_text_simple  # 从第4章导入GPTModel
import torch
import tiktoken
from ch02.GPTDatasetV1 import create_dataloader_v1  # 从第2章导入create_dataloader_v1

GPT_CONFIG_124M = {  # GPT配置字典
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # A 将上下文长度从1024缩短到256词元
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # B 可能且常见的是将dropout设置为0。
    "qkv_bias": False  # QKV偏置
}


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


if __name__ == '__main__':
    torch.manual_seed(123)  # 设置随机种子
    model = GPTModel(GPT_CONFIG_124M)  # 使用配置初始化模型
    model.eval()  # 将模型设置为评估模式


    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的分词器编码



    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves"]
                           [40, 1107, 588]])  # ["I really like"]
    # 匹配这些输入，'targets'包含我们希望模型生成的词元ID
    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you"]
                            [107, 588, 11311]])  # [" really like chocolate"]
    with torch.no_grad():  # 禁用梯度跟踪，因为我们尚未训练
        logits = model(inputs)  # 将输入送入模型，计算logit向量
        probas = torch.softmax(logits, dim=-1)  # 每个词元在词汇表中的概率
    # print(probas.shape)  # 打印概率张量的维度
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)  # 应用argmax函数获得词元ID
    # print("Token IDs:\n", token_ids)  # 打印词元ID
    # print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")  # 打印第一批次的目标词元
    # print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")  # 打印第一批次的输出词元



    text_idx = 0  # 文本索引0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]  # 计算目标词元的softmax概率
    # print("Text 1:", target_probas_1)  # 打印文本1的概率

    text_idx = 1  # 文本索引1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]  # 计算目标词元的softmax概率
    # print("Text 2:", target_probas_2)  # 打印文本2的概率

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))  # 计算对数概率
    # print(log_probas)  # 打印对数概率

    avg_log_probas = torch.mean(log_probas)  # 计算平均对数概率
    # print(avg_log_probas)  # 打印平均对数概率

    neg_avg_log_probas = avg_log_probas * -1  # 计算负平均对数概率
    # print(neg_avg_log_probas)  # 打印负平均对数概率

    logits_flat = logits.flatten(0, 1)  # 将logits展平
    targets_flat = targets.flatten()  # 将目标展平
    # print("Flattened logits:", logits_flat.shape)  # 打印展平后的logits形状
    # print("Flattened targets:", targets_flat.shape)  # 打印展平后的目标形状
    # print(logits_flat)
    # print(targets_flat)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)  # 计算交叉熵损失
    # print(loss)  # 打印损失

    file_path = "the-verdict.txt"  # 文件路径
    with open(file_path, "r", encoding="utf-8") as file:  # 以读模式打开文件
        text_data = file.read()  # 读取文件内容

    # total_characters = len(text_data)  # 计算总字符数
    # total_tokens = len(tokenizer.encode(text_data))  # 计算总词元数
    # print("Characters:", total_characters)  # 打印字符数
    # print("Tokens:", total_tokens)  # 打印词元数

    train_ratio = 0.90  # 训练集比例
    split_idx = int(train_ratio * len(text_data))  # 计算分割索引
    train_data = text_data[:split_idx]  # 获取训练数据
    val_data = text_data[split_idx:]  # 获取验证数据


    torch.manual_seed(123)  # 设置随机种子
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
    model.to(device)  # 将模型移动到设备上
    with torch.no_grad():  # 禁用梯度计算以提高效率
        train_loss = calc_loss_loader(train_loader, model, device)  # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device)  # 计算验证集损失
    print("Training loss:", train_loss)  # 打印训练集损失
    print("Validation loss:", val_loss)  # 打印验证集损失

