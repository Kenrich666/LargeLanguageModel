from importlib.metadata import version
import tiktoken

# print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = "Akwirw ier"
integers = tokenizer.encode(text)
print(integers)

strings = tokenizer.decode(integers)
print(strings)