import re


# 一个完整的 tokenizer 类
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab  # 将表存储为类属性，便于在编码和解码方法中访问
        self.int_to_str = {i: s for s, i in vocab.items()}  # 创建一个逆转录表，将 token ID 映射回原始文本 token

    # 将输入文本处理为 token ID
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 将 token ID 重新转换为文本
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 替换指定标记点符号前面的空格
        return text
