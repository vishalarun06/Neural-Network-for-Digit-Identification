import numpy as np

class Loss:
    """
    A blueprint for all loss functions.
    """
    def forward(self, predictions, targets):
        raise NotImplementedError
        
    def backward(self, predictions, targets):
        raise NotImplementedError

## Mean-Squared Error
class MeanSquaredError(Loss):
    """
    Calculates the average squared difference between predictions and targets.
    """

    def forward(self, predictions, targets):
        return np.mean(np.power(predictions - targets, 2))


    def backward(self, predictions, targets):

        m = targets.size

        return 2 * (predictions - targets) / m

    

