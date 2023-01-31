import numpy as np
from layer_dense import Layer_Dense

class Layer_Output(Layer_Dense):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)
    
    def activation_function(self, inputs):
        '''Softmax activation function'''
        # Reduce the size of the inputs to avoid overflow after exponentiation
        inputs = inputs - np.max(inputs)
        # Calculate the exponential of each input to handle negative values
        exp_values = np.exp(inputs)
        # Normalize the values to get the probabilities
        normalized_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return normalized_values
    
    def activation_function_derivative(self, outputs):
        '''Derivative of softmax activation function'''
        return outputs * (1 - outputs)