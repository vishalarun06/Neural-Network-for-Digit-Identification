import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """ This will be our activation function """
    # Prevents overflow errors in exp by limiting extreme values
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    """
    The derivative of the sigmoid function. 
    Required for backpropagation to calculate how much to tweak the weights.
    """
    s = sigmoid(z)
    return s * (1 - s)

class NeuralNetwork221:
    def __init__(self):
        """
        This will create the layers of our network
        - Layer 0 (Input): 2 neurons
        - Layer 1 (Hidden): 2 neurons
        - Layer 2 (Output): 1 neuron
        """
        self.W1 = np.random.randn(4,2) * 0.1
        self.B1 = np.zeros((4,1))

        self.W2 = np.random.randn(1,4) * 0.1
        self.B2 = np.zeros((1,1))

    def forward_prop(self, X):
        """
        This will take an input and propagate forward through the network to give an output
        """
        # Through Layer 1
        self.Z1 = np.dot(self.W1,X) + self.B1
        self.A1 = sigmoid(self.Z1) # Sigmoid activation function

        # Through Layer 2
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = sigmoid(self.Z2)

        return self.A2
    
    def back_prop(self, X, Y):
        """
        This will calculate the gradients and determine how we adjust the weights and biases
        """
        m = X.shape[1] # Batch size

        delta_out = (self.A2 - Y) * sigmoid_deriv(self.Z2)
        dW2 =  np.dot(delta_out, self.A1.T)
        dB2 =  np.sum(delta_out, axis=1, keepdims=True)

        delta_hidden = np.dot(self.W2.T, delta_out) * sigmoid_deriv(self.Z1)
        dW1 = np.dot(delta_hidden, X.T)
        dB1 = np.sum(delta_hidden, axis=1, keepdims=True)

        return dW1, dB1, dW2, dB2
    
    def update_params(self, dW1, dB1, dW2, dB2, learning_rate):
        """
        Tweaks the weights and biases by the gradients proportional to learning rate
        in the direction that reduces the error.
        """
        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * dB1
        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * dB2

    def train_sgd(self, X, Y, iterations=10000, learning_rate=1.0):
        """
        Trains the network using Stochastic Gradient Descent (one example at a time)
        to prevent perfectly symmetric batches from canceling out the gradients.
        """
        print("Starting SGD training...")
        
        # Get the number of examples in our dataset (4 for XOR)
        num_examples = X.shape[1] 
        
        for i in range(iterations):
            
            # Loop through each example one by one
            for j in range(num_examples):
                
                # Slice out a single column (keep it 2D so matrix math doesn't break)
                x_single = X[:, j:j+1]
                y_single = Y[:, j:j+1]

                # Forward Pass for JUST this one example
                self.forward_prop(x_single)

                # Back Propagation for JUST this one example
                dW1, dB1, dW2, dB2 = self.back_prop(x_single, y_single)

                # Update the weights instantly before looking at the next example
                self.update_params(dW1, dB1, dW2, dB2, learning_rate)

        print("Training complete!")

        print("Training complete!")

nn = NeuralNetwork221()

# 2. The XOR Training Data
# Row 1: The first inputs (0, 0, 1, 1)
# Row 2: The second inputs (0, 1, 0, 1)
X = np.array([[0, 0, 1, 1], 
              [0, 1, 0, 1]])

# The target answers corresponding to each column above
Y = np.array([[0, 1, 1, 0]])

# 3. Train the network! (XOR usually takes a few thousand epochs to click)
nn.train_sgd(X, Y, iterations=10000, learning_rate=1)

# 4. Print the final predictions to see if it learned the gate
print("\n--- Final Network Predictions ---")
predictions = nn.forward_prop(X)
for i in range(4):
    print(f"Inputs: {X[:, i]} | Target: {Y[0, i]} | Network Guessed: {predictions[0, i]:.4f}")
