import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons) - 0.5
        self.biases = np.zeros((1, n_neurons))

    def set_weights(self, weights):
        '''Set the weights of the layer'''
        self.weights = weights
    
    def set_biases(self, biases):
        '''Set the biases of the layer'''
        self.biases = biases
    
    def forward(self, inputs):
        '''ouput = inputs * weights + biases'''
        self.inputs = inputs
        output = np.dot(self.inputs, self.weights) + self.biases
        self.output = output
        return self.output
    
    def backward(self, output_error, learning_rate):
        '''Backpropagation'''
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.inputs.T, output_error)
        self.weights -= learning_rate * weights_error
        #self.biases -= learning_rate * np.mean(output_error)
        return input_error