import numpy as np

class Sequential:
    """
    This will hold the key architecture of the neural network.
    """
    def __init__(self, layers):
        self.Layers = layers

    def forward(self, inputs):
        """
        This method will iterate through each layer and return its fianl prediction
        """
        temp = inputs

        for layer in self.Layers:
            temp = layer.forward(temp)

        return temp

    
    def backward(self, grad_output):
        """
        This method will iterate backwards through the layers to calculate the gradients and adjustments
        """
        temp_grad = grad_output

        for layer in reversed(self.Layers):
            temp_grad = layer.backward(temp_grad)

        return temp_grad

    




    