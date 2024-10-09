import torch

from ch04.GPTModel import GPTModel, generate_text_simple
from ch05.Train import text_to_token_ids, token_ids_to_text
from ch06.Config import CHOOSE_MODEL, BASE_CONFIG, load_weights_into_gpt
from ch06.CreateDataLoaders import tokenizer
from ch06.gpt_download import download_and_load_gpt2

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
# print(inputs)
inputs = torch.tensor(inputs).unsqueeze(0)
# print("Inputs:", inputs)
# print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)

# with torch.no_grad():
#     outputs = model(inputs)
# print("Outputs:\n", outputs)
# print("Outputs dimensions:", outputs.shape)  # shape: (batch_size, num_tokens, num_classes)
#
# print("Last output token:", outputs[:, -1, :])
