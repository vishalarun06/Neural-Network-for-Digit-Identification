import numpy as np
from layers import Layer

class ReLU(Layer):
    """ 
    Defines the activation function
    """

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        # Returns the input if it's > 0, otherwise returns 0
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        # The derivative of ReLU is 1 if input > 0, and 0 otherwise.
        return grad_output * (self.inputs > 0)
    