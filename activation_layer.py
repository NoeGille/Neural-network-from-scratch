import numpy as np

class ActivationLayer():

    def __init__(self, activation, activatoin_derivative):
        self.activation = activation
        self.activation_derivative = activatoin_derivative
        self.weights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return self.activation_derivative(self.inputs) * output_error