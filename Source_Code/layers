import numpy as np

class Layer:

    """ 
    A base class, that guarantees that each layer has weights, biases and gradients
    This ensure that we can propagate both forwards and backwards through each layer.
    """

    def __init__(self):
        self.params = {} # Stores the weights and biases
        self.gradients = {} # Stores the gradients, dW and db

    def forward(self, inputs):
        ## Takes in inputs, performs a mathematical operation and then outputs to another layer

        raise NotImplementedError("Every specific layer must implement its own forward pass.")

    def backward(self, grad_outputs):
        ## Takes the error gradient from the layer ahead of it, calculates its own 
        ## parameter's gradient and calculates the error for the layer behind it

        raise NotImplementedError("Every specific layer must implement its own backward pass.")

class Dense(Layer):
    """ 
    A fully connected layer being implemented
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        self.params['W'] = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size) # The weights key in the dictionary
        self.params['b'] = np.zeros((output_size, 1)) # The biases key in the dictionary

        self.inputs = None

    def forward(self, inputs):

        self.inputs = inputs

        return np.dot(self.params['W'], inputs) + self.params['b']

    def backward(self, grad_output):

        m = self.inputs.shape[1] # This is the number of examples in a batch, can use this for averaging out gradients

        self.gradients['W'] = (1/m) * np.dot(grad_output, self.inputs.T)
        self.gradients['b'] = (1/m) * np.sum(grad_output, axis=1, keepdims=True)

        ## Calculates the error to pass back to the previous layer
        grad_input = np.dot(self.params['W'].T, grad_output)
        
        return grad_input
    


