import numpy as np

LEARNING_RATE = 0.1

class Neural_Network:
    def __init__(self) -> None:
        self.layers = []

    def loss(self, y_true, y_pred):
        '''MSE loss function'''
        print(y_pred)
        return np.mean(np.square(y_true - y_pred),axis=0)
    
    def loss_derivative(self, y_true, y_pred):
        '''Derivative of MSE loss function'''
        return 2 * (y_pred - y_true) / y_true.size
    
    def add_layer(self, layer):
        '''Add a layer to the network'''
        self.layers.append(layer)

    def train(self, X_train, y_train):
        '''Train the network on a sample'''
        # Forward propagation
        output = X_train
        for layer in self.layers:
            output = layer.forward(output)
        output_error = self.loss(y_train, output)
        print(output_error)
        print("output_error_shape", output_error.shape)
        # Backpropagation
        # output_error = self.loss_derivative(y_train, output)
        for layer in reversed(self.layers):
           
            output_error = layer.backward(output_error, LEARNING_RATE)
    def predict(self, X_test):
        '''Predict the output of the network'''
        output = X_test
        for layer in self.layers:
            output = layer.forward(output)
        return output

