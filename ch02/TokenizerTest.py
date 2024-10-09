import re

from ch02.SimpleTokenizerV1 import SimpleTokenizerV1
from ch02.SimpleTokenizerV2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# 切分token
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# print(len(preprocessed))
# print(preprocessed[:30])

# token --> token_ID

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
# # print(vocab_size)
# vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# V1

# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

# V2

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))