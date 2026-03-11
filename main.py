import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl, certifi, platform
import MCDropout as mcd

if platform.system() == "Darwin":
    print("Code running on MacOS. Fixing SSL Certificate.")
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    print("SSL Certificate fixed!\n")

    device = 'mps'
    print("MPS available:", torch.backends.mps.is_available())
    print("Running on:", device)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

labels_map = {label : str(label) for label in range(10)}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

model = mcd.neural_network_model(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("\nTraining the model...\n")

epochs = 1
for epoch in range(epochs):
    model.train()

    for X, y in train_data_loader:
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

for img, label in test_data:
    if label not in checked_labels:
        sample_img = img.unsqueeze(0).to(device)

        mean_pred, epistemic, aleatoric = mcd.monte_carlo_dropout(model, sample_img, 100, train)
        pred_class = mean_pred.argmax(dim=1).item()

        print("Predicted class:", pred_class)
        print("Real label:", label)
        print("Epistemic uncertainty:", epistemic.item())
        print("Aleatoric uncertainty:", aleatoric.item())

        print(f"Accuracy: {mcd.accuracy(model, test_data_loader, device):.2f}%")

        train = True
        checked_labels.add(label)

        print("\n-----------\n")