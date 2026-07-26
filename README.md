Neural Network Framework from Scratch (NumPy Only)
Most of the time, building a neural network is just letting a library like PyTorch do the heavy lifting. I built this project to look inside that black box.

This is a fully modular, object-oriented deep learning framework written entirely in Python and NumPy. I wanted to understand how to do all the matrix calculus, backpropagation mechanics, and software engineering principles from scratch.

**Interesting Features:** 
It's completely modular: Instead of writing one massive script, I broke the network down into interchangeable pieces (Layers, Activations, Losses, and Optimisers). You can snap them together to build any network architecture you want.

I did the math by hand: There is no automatic differentiation here. Every forward pass and backpropagation gradient is derived and implemented using pure NumPy matrix operations.

Advanced optimisers: Basic Gradient Descent struggles with noisy data, so I built an Adam optimizer from scratch. It tracks momentum and variance to dynamically adjust the learning rate for every single weight.

**Organisation**
I structured this project to mirror professional engineering environments, keeping the core logic cleanly separated:

Source_Code/layers.py: The blueprint for the network's layers (like Dense layers) that hold the weights and calculate local gradients.

Source_Code/activations.py: The non-linear transformations (ReLU) that give the network its predictive power.

Source_Code/losses.py: The functions (Mean Squared Error) that calculate how wrong the network is and kick off the backpropagation chain.

Source_Code/optimizers.py: The mechanics (Gradient Descent, Adam) that actually update the weights based on the calculated gradients.

Source_Code/model.py: The Sequential container that loops through the layers, passing data forward and error signals backward.
