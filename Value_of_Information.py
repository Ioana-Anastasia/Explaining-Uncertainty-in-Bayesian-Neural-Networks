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
        nn.Flatten(),#convert to 1D
        nn.Linear(64, 64), #classify according to weights
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 10)
    ).to(device)
    return model

# run monte carlo dropout with VOI
def monte_carlo_dropout_with_voi(model, X, n_samples, train):

    if not train:
        model.train()

    predictions = []

    with torch.no_grad(): #inference validation and weight update
        for _ in range(n_samples):
            out = functional.softmax(model(X), dim=1)
            predictions.append(out)

    # get predictions
    predictions = torch.stack(predictions, dim =0)

    # compute the mean of the predictions
    mean_pred = predictions.mean(dim=0)

    # and the expected and predicted entropy
    predicted_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2).mean(dim=0)

    #and get the value of information
    voi = expected_entropy - predicted_entropy

    return voi, mean_pred