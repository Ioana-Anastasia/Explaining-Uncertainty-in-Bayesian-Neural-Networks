import torch
import torch.nn as nn
import torch.nn.functional as functional

# define the model/neural network
def neural_network_model(device):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3), # feature extraction
        nn.ReLU(), #break linearity
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2), #introduce variance
        nn.Dropout2d(0.25), # deactivate data
        nn.AdaptiveAvgPool2d(1), #fix output size
        nn.Flatten(), #convert to 1D
        nn.Linear(64, 64), #classify according to weights
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 10)
    ).to(device)
    return model

# function to get the accuracy of monte carlo dropout
def accuracy(model, data, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): #inference validation and weight update
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) #element-wise maximum - return new tensor with maximum of each
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # computed by dividing the number of correctly predicted values by the total amount
    return 100 * correct / total

# run monte carlo dropout
def monte_carlo_dropout(model, X, n_samples, train):

    if not train:
        model.train()

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = functional.softmax(model(X), dim=1) # predictions to probabilities
            predictions.append(out)

    # get predictions
    predictions = torch.stack(predictions, dim =0)

    # compute the mean of the predictions
    mean_pred = predictions.mean(dim=0)

    # and the expected and predicted entropy
    predicted_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2).mean(dim=0)

    # perform uncertainty decomposition into epistemic and aleatoric
    epistemic = expected_entropy - predicted_entropy
    aleatoric = expected_entropy

    return mean_pred, epistemic, aleatoric