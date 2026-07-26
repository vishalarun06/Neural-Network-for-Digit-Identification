import numpy as np

class grad_descent:
    """
    Performs the gradient descent for each step
    """
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, model):

        for layer in model.layers:

            if layer.params:

                layer.params['W'] -= self.lr * layer.gradients['W'] # Update weights
                layer.params['b'] -= self.lr * layer.grads['b'] # Update biases

class Adam:
    """
    Adaptive Moment Estimation (Adam).
    Instead of a fixed learning rate for every weight, Adam gives EVERY SINGLE WEIGHT 
    its own custom learning rate based on how it behaved in the past.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Dictionaries to store the "memory" of past gradients for each layer
        self.m = {} # First moment (momentum)
        self.v = {} # Second moment (velocity/variance)
        self.t = 0  # Time step

    def step(self, model):
        self.t += 1
        
        for layer_idx, layer in enumerate(model.layers):
            if not layer.params:
                continue # Skip activation layers
                
            # If this is the first time seeing this layer, create its memory banks
            if layer_idx not in self.m:
                self.m[layer_idx] = {'W': np.zeros_like(layer.params['W']), 
                                     'b': np.zeros_like(layer.params['b'])}
                self.v[layer_idx] = {'W': np.zeros_like(layer.params['W']), 
                                     'b': np.zeros_like(layer.params['b'])}

            for param_name in ['W', 'b']:
                grad = layer.grads[param_name]
                
                # 1. Update biased first moment estimate (Momentum)
                self.m[layer_idx][param_name] = self.beta1 * self.m[layer_idx][param_name] + (1 - self.beta1) * grad
                
                # 2. Update biased second raw moment estimate (RMSprop)
                self.v[layer_idx][param_name] = self.beta2 * self.v[layer_idx][param_name] + (1 - self.beta2) * (grad ** 2)
                
                # 3. Compute bias-corrected estimates
                m_hat = self.m[layer_idx][param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[layer_idx][param_name] / (1 - self.beta2 ** self.t)
                
                # 4. Update the actual parameters
                layer.params[param_name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
