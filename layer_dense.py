import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def set_weights(self, weights):
        self.weights = weights
    
    def set_biases(self, biases):
        self.biases = biases
    
    def forward(self, inputs):
        output = self.activation_function(np.dot(inputs, self.weights) + self.biases)
        return output

    def activation_function(self, inputs):
        '''Activation ReLU'''
        output = np.maximum(0, inputs)
        return output
    
    def activation_function_derivative(self, outputs):
        '''Derivative of ReLU activation function'''
        return 0 if outputs <= 0 else 1