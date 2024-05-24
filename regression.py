import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    learning_rate = 1e-3 
    num_epochs = 1000  
    input_features = X.shape[1]
    output_features = y.shape[1] if len(y.shape) > 1 else 1
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previos_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if abs(previos_loss - loss.item()) < 1e-6:
            break
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        previos_loss = loss.item()

    return model, loss

