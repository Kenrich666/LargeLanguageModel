import torch

from ch06.Calculate import calc_accuracy_loader, calc_loss_loader
from ch06.CreatePyTorchDataLoaders import train_loader, val_loader, test_loader
from ch06.LoadModel import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

torch.manual_seed(123)
# train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
# val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
# test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
#
# print(f"Training accuracy: {train_accuracy * 100:.2f}%")
# print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
# print(f"Test accuracy: {test_accuracy * 100:.2f}%")

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")
