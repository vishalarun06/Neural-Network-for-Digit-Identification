import numpy as np

from Source_Code.layers import Dense
from Source_Code.activations import ReLU
from Source_Code.losses import MeanSquaredError
from Source_Code.optimisers import grad_descent
from Source_Code.model import Sequential

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    ReLU()
])

loss_fn = MeanSquaredError()
optimiser = grad_descent(learning_rate=0.01)

def train(X_batch, Y_batch):

    predictions = model.forward(X_batch)

    loss_value = loss_fn.forward(predictions, Y_batch)

    initial_gradient = loss_fn.backward(predictions, Y_batch)

    model.backward(initial_gradient)

    optimiser.step(model)

    return loss_value


