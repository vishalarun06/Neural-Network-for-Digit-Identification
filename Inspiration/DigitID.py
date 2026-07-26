import numpy as np
from sklearn.datasets import fetch_openml
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps



def sigmoid(z):
    """
    The Sigmoid squishification function [00:10:29].
    Takes any real number and squishes it into a range between 0 and 1.
    """
    
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    """
    The derivative of the sigmoid function. 
    Required for backpropagation to calculate how much to tweak the weights.
    """
    s = sigmoid(z)
    return s * (1 - s)

def one_hot_encode(Y, num_classes=10):
    """
    Converts a number, n, into a column vector 
    where the nth index is 1 and all others are 0.
    """
    one_hot = np.zeros((num_classes, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot


class NeuralNetwork3B1B:
    def __init__(self):
        """
        - Layer 0 (Input): 784 neurons (28x28 pixels)
        - Layer 1 (Hidden): 16 neurons
        - Layer 2 (Hidden): 16 neurons
        - Layer 3 (Output): 10 neurons (Digits 0-9)
        """
        
        
        # 1st Transition (784 -> 16)
        self.W1 = np.random.randn(16, 784) * 0.1
        self.b1 = np.zeros((16, 1))
        
        # 2nd Transition (16 -> 16)
        self.W2 = np.random.randn(16, 16) * 0.1
        self.b2 = np.zeros((16, 1))
        
        # 3rd Transition (16 -> 10)
        self.W3 = np.random.randn(10, 16) * 0.1
        self.b3 = np.zeros((10, 1))

    def forward_prop(self, X):
        """
        Passes the image data forward through the network using the matrix multiplication
        """
        # Layer 1
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        # Layer 3 (Output)
        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        self.A3 = sigmoid(self.Z3)
        
        return self.A3

    def backward_prop(self, X, Y):
        """
        Calculates the gradients (how much to adjust weights and biases).
        """
        m = X.shape[1] # Number of examples in this batch
        Y_one_hot = one_hot_encode(Y)
        
        # Calculate error at the output layer
        # Difference between our network's guess (A3) and the real answer (Y_one_hot)
        delta_out = (self.A3 - Y_one_hot) * sigmoid_deriv(self.Z3)
        
        # Gradients for Layer 3 parameters
        dW3 = (1/m) * np.dot(delta_out, self.A2.T)
        db3 = (1/m) * np.sum(delta_out, axis=1, keepdims=True)
        
        # Propagate the error backward to Layer 2
        delta_2 = np.dot(self.W3.T, delta_out) * sigmoid_deriv(self.Z2)
        dW2 = (1/m) * np.dot(delta_2, self.A1.T)
        db2 = (1/m) * np.sum(delta_2, axis=1, keepdims=True)
        
        # Propagate the error backward to Layer 1
        delta_1 = np.dot(self.W2.T, delta_2) * sigmoid_deriv(self.Z1)
        dW1 = (1/m) * np.dot(delta_1, X.T)
        db1 = (1/m) * np.sum(delta_1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        """
        Adjusts the weights and biases proportional to the learning rate
        """
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def get_predictions(self, A3):
        """
        Finds the most active neuron in the output layer.
        """
        return np.argmax(A3, axis=0)

    def get_accuracy(self, predictions, Y):
        """
        Calculates what percentage of the predictions were correct.
        """
        return np.sum(predictions == Y) / Y.size

    def train(self, X, Y, iterations=1000, learning_rate=0.5):
        """
        The main training loop. Repeats forward pass, backward pass, and updates.
        """
        print("Starting training...")
        for i in range(iterations):
            # 1. Guess the answers (Forward pass)
            A3 = self.forward_prop(X)
            
            # 2. See how wrong we were and calculate adjustments (Backward pass)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(X, Y)
            
            # 3. Adjust the weight and biases (Update parameters)
            self.update_params(dW1, db1, dW2, db2, dW3, db3, learning_rate)
            
            # Print progress every 100 steps
            if i % 100 == 0:
                predictions = self.get_predictions(A3)
                accuracy = self.get_accuracy(predictions, Y)
                print(f"Iteration {i:4d} | Accuracy: {accuracy*100:.2f}%")
        print("Training complete!")

class DigitDrawer:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Draw a Digit")
        
        # Canvas setup
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='black')
        self.canvas.pack()
        
        # Internal image to keep track of drawings
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        tk.Button(self.root, text="Predict", command=self.predict).pack()
        tk.Button(self.root, text="Clear", command=self.clear).pack()
        
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 12), (event.y - 12)
        x2, y2 = (event.x + 12), (event.y + 12)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # 1. Resize to 28x28
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 2. Convert to numpy array, flatten, and normalize
        img_array = np.array(img).reshape(784, 1) / 255.0
        
        # 3. Use your existing model
        output = self.model.forward_prop(img_array)
        prediction = np.argmax(output)
        print(f"Neural Network Prediction: {prediction}")

    def learn_from_feedback(self, X_input, true_label, learning_rate=0.01):
        # 1. Forward propagate to get current state
        A3 = self.forward_prop(X_input)
        
        # 2. Perform a small training step
        Y_target = np.zeros((10, 1))
        Y_target[true_label] = 1
        
        # 3. Backward prop based on this specific example
        dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(X_input, np.array([true_label]))
        self.update_params(dW1, db1, dW2, db2, dW3, db3, learning_rate)



if __name__ == "__main__":
    # 1. Training the model on MNIST data
    nn = NeuralNetwork3B1B()
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X_real = mnist.data.to_numpy().T / 255.0
    Y_real = mnist.target.to_numpy().astype(int)
    
    print("Training model...")
    nn.train(X_real, Y_real, iterations=10000, learning_rate=0.5)
    
    # 2. Launch the drawing app
    print("Launching drawing interface...")
    DigitDrawer(nn)
