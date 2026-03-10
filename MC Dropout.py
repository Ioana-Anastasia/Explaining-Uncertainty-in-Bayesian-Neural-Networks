import main as mn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import platform

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
            out = F.softmax(model(X), dim=1)
            predictions.append(out)

    predictions = torch.stack(predictions, dim =0)

    mean_pred = predictions.mean(dim=0)

    total_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2).mean(dim=0)

    epistemic = total_entropy - expected_entropy
    aleatoric = expected_entropy

    return mean_pred, epistemic, aleatoric


def main():

    if platform.system() == "Darwin":
        device = 'mps'
        print("MPS available:", torch.backends.mps.is_available())
        print("Running on:", device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    train_data = DataLoader(mn.train_data, batch_size=64, shuffle=True)
    test_data = DataLoader(mn.test_data, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining the model...\n")

    epochs = 15
    for epoch in range(epochs):
        model.train()

        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("\nModel training completed.\n")
    print("\nEvaluating the model...\n")

    train = False
    checked_labels = set()

    for img, label in mn.test_data:
        if label not in checked_labels:
            sample_img = img.unsqueeze(0).to(device)

    # for label in range(10):
    #     sample_img, label = mn.test_data[label]
    #     sample_img = sample_img.unsqueeze(0).to(device)

            mean_pred, epistemic, aleatoric = monte_carlo_dropout(model, sample_img, 100, train)
            pred_class = mean_pred.argmax(dim=1).item()

            print("Predicted class:", pred_class)
            print("Real label:", label)
            print("Epistemic uncertainty:", epistemic.item())
            print("Aleatoric uncertainty:", aleatoric.item())

            print(f"Accuracy: {accuracy(model, test_data, device):.2f}%")

            train = True
            checked_labels.add(label)

            print("\n-----------\n")

if __name__ == "__main__":
    main()




