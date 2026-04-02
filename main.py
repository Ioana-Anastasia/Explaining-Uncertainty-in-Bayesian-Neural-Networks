import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl
import certifi
import platform
import Monte_Carlo_Dropout as mcd
import Value_of_Information as voi
import os

# set the running device and prevent operating system possible conflicts
if platform.system() == "Darwin":
    print("Code running on MacOS. Fixing SSL Certificate.")
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    print("SSL Certificate fixed!\n")

    device = 'mps'
    print("MPS available:", torch.backends.mps.is_available())
    print("Running on:", device)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {platform.system()}, using {device}.")

# create 2 .txt files to store results - if they already exist, then overwrite them
if not os.path.isfile("mcd_results.txt") and not os.path.isfile("voi_results.txt"):
    mcd_results = open("mcd_results.txt", 'x')
    voi_results = open("voi_results.txt", 'x')

else:
    with open("mcd_results.txt", "r+") as file_mcd:
        file_mcd.seek(0)
        file_mcd.truncate()

    with open("voi_results.txt", "r+") as file_voi:
        file_voi.seek(0)
        file_voi.truncate()

# define and ste transformers
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# get train data
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# get test data
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# transform the data in DataLoaders - required for working with tensors
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# print a 3x3 matrix of 9 random elements from the dataset - for visualization purposes
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

# get the model from the Monte Carlo file (identical to the mode in the Value of Information file)
model = mcd.neural_network_model(device)

# set the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("\nTraining the model...\n")

# train the model on 15 epochs and 100 inputs
epochs = 15
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

print("\n\nMONTE CARLO DROPOUT:\n\n")

# evaluate the Monte Carlo Dropout model
train = False # set to false initially
checked_labels = set() #$ keep track of already printed labels so we can evaluate all
file_mcd = open("mcd_results.txt", "a")

for img, label in test_data:
    if label not in checked_labels:
        sample_img = img.unsqueeze(0).to(device)

        mean_pred, epistemic, aleatoric = mcd.monte_carlo_dropout(model, sample_img, 100, train)
        pred_class = mean_pred.argmax(dim=1).item()

        print("Predicted label:", pred_class)
        print("Real label:", label)
        print("Epistemic uncertainty:", epistemic.item())
        print("Aleatoric uncertainty:", aleatoric.item())

        file_mcd.write(f"Predicted label: {pred_class}, "
                       f"Real label: {label}, "
                       f"Epistemic uncertainty: {epistemic.item()}, "
                       f"Aleatoric uncertainty: {aleatoric.item()}\n")

        print(f"Accuracy: {mcd.accuracy(model, test_data_loader, device):.2f}%")

        train = True # set to true - model has been trained - we don't have to train it again for VOI
        checked_labels.add(label) # set to make sure we have no duplicate in labels

        print("\n-----------\n")

file_mcd.close()

print("\n\nVALUE OF INFORMATION:\n\n")

# evaluate Value of Information

checked_labels = set() # reinitialise the set of labels - same purpose as for MCD
file_voi = open("voi_results.txt", "a")

for img, label in test_data:
    if label not in checked_labels:
        sample_img = img.unsqueeze(0).to(device)

        voi_value, mean_pred = voi.monte_carlo_dropout_with_voi(model, sample_img, 100, train)
        pred_class = mean_pred.argmax(dim=1).item()

        print("Predicted class:", pred_class)
        print("Real label:", label)
        print("Value of information:", voi_value.item())

        file_voi.write(f"Predicted class: {pred_class}, "
                       f"Real label: {label}, "
                       f"Value of information:, {voi_value.item()}\n")

        checked_labels.add(label)

        print("\n-----------\n")

file_voi.close()
