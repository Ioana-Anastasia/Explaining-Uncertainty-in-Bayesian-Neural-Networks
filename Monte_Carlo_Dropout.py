import torch
import torch.nn as nn
import torch.nn.functional as functional

def neural_network_model(device):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 10)
    ).to(device)
    return model

def accuracy(model, data, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def monte_carlo_dropout(model, X, n_samples, train):

    if not train:
        model.train()

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = functional.softmax(model(X), dim=1)
            predictions.append(out)

    predictions = torch.stack(predictions, dim =0)

    mean_pred = predictions.mean(dim=0)

    predicted_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2).mean(dim=0)

    epistemic = expected_entropy - predicted_entropy
    aleatoric = expected_entropy

    return mean_pred, epistemic, aleatoric