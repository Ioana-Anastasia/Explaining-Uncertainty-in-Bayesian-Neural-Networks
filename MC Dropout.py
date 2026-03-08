import main as mn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import platform

def monte_carlo_dropout(model, X, n_samples):
    model.train()

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = F.softmax(model(X), dim=1)
            predictions.append(out.unsqueeze(0))

        predictions = torch.cat(predictions, dim=0)
        means_predict = predictions.mean(dim=0)
        std_predict = predictions.std(dim=0)
        return means_predict, std_predict


def main():

    if platform.system() == "Darwin":
        device = 'mps'
        print("MPS available:", torch.backends.mps.is_available())
        print("Running on:", device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    ).to(device)

    train_data = DataLoader(mn.train_data, batch_size=64, shuffle=True)
    test_data = DataLoader(mn.test_data, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining the model...\n")

    epochs = 20
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

    sample_img, label = mn.test_data[0]
    sample_img = sample_img.unsqueeze(0).to(device)

    mean_pred, std_pred = monte_carlo_dropout(model, sample_img, 100)
    pred_class = mean_pred.argmax(dim=1).item()

    print("Predicted class:", pred_class)
    print("Prediction uncertainty (std per class):", std_pred.cpu().numpy())

if __name__ == "__main__":
    main()




