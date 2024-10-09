import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对整个文本进行标记
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口将书分成max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集中的总行数
    def __len__(self):
        return len(self.input_ids)

    # 从数据集中返回一行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # drop_last=True 如果最后一批批次短于指定的batch_size，则删除该批次，以防止训练期间出现损失高峰
        drop_last=drop_last,
        # 用于预处理的CPU进程数
        num_workers=0
    )

    return dataloader


if __name__ == '__main__':
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # dataloader = create_dataloader_v1(
    #     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(first_batch)
    #
    # second_batch = next(data_iter)
    # print(second_batch)

    # dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)
    #
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print("Inputs:\n", inputs)
    # print("\nTargets:\n", targets)


    vocab_size = 50257  # 词汇表大小
    output_dim = 256  # 输出维度
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 实例化词元嵌入层

    max_length = 4  # 最大长度
    dataloader = create_dataloader_v1(  # 创建数据加载器
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)  # 将数据加载器转换为Python迭代器
    inputs, targets = next(data_iter)  # 获取下一个批次的数据
    # print("Token IDs:\n", inputs)  # 打印词元ID
    # print("\nInputs shape:\n", inputs.shape)  # 打印输入的形状

    token_embeddings = token_embedding_layer(inputs)  # 获取词元嵌入
    # print(token_embeddings.shape)  # 打印词元嵌入的形状
    # 上述打印函数返回以下内容：
    # torch.Size([8, 4, 256])

    context_length = max_length  # 上下文长度等于最大长度
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # 创建位置嵌入层
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # 获取位置嵌入
    # print(pos_embeddings.shape)  # 打印位置嵌入的形状

    input_embeddings = token_embeddings + pos_embeddings  # 将位置嵌入与词元嵌入相加
    print(input_embeddings.shape)  # 打印输入嵌入的形状




