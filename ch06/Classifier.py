import tiktoken
import torch

from ch04.GPTModel import GPTModel
from ch06.CreateDataLoaders import train_dataset

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


if __name__ == '__main__':

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(CONFIG)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=CONFIG["emb_dim"], out_features=num_classes)

    model_state_dict = torch.load("review_classifier.pth", weights_only=True)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    text_1 = (
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))
