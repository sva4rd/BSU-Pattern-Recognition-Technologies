import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from FTML import FTML
from Perceptron import Perceptron


def train_model(model, train_loader, test_loader, optimizer, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.view(data.shape[0], -1))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.view(data.shape[0], -1))
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
        if best_accuracy < 100. * correct / len(test_loader.dataset):
            best_accuracy = 100. * correct / len(test_loader.dataset)

    return best_accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

input_size = 784
output_size = 10
results = {}
for hidden_size1 in [5, 10, 15, 20, 25]:
    for hidden_size2 in [5, 10, 15, 20, 25]:
        model = Perceptron(input_size, hidden_size1, hidden_size2, output_size)
        optimizer = FTML(model.parameters(), lr=0.0025)
        print(f"Training model with {hidden_size1} and {hidden_size2} neurons in the hidden layers.")
        accuracy = train_model(model, train_loader, test_loader, optimizer)
        results[(hidden_size1, hidden_size2)] = accuracy

hidden_sizes = [5, 10, 15, 20, 25]
data = {hs2: [] for hs2 in hidden_sizes}
index = []

for hs1 in hidden_sizes:
    index.append(hs1)
    for hs2 in hidden_sizes:
        data[hs2].append(results[(hs1, hs2)])

df = pd.DataFrame(data, index=index, columns=hidden_sizes)
print(df)
