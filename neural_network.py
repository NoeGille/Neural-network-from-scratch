import numpy as np
import matplotlib.pyplot as plt
from layer_dense import Layer_Dense
from output_layer import Layer_Output

class Neural_Network:
    def __init__(self, inputs) -> None:
        self.input_layer = inputs
        self.layer1 = Layer_Dense(2, 5)
        self.layer2 = Layer_Dense(5, 5)
        self.output_layer = Layer_Output(5, 3)

    def loss_categorical_cross_entropy(self, y_valid, y_pred):
        '''Loss is a measure of how wrong the model is.
        We want to minimize the loss.'''
        # Clip data to prevent division by 0 which causes inf values with numpy
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate the loss
        loss = np.mean(-np.log(y_pred_clipped[range(len(y_pred)), y_valid]))
        return loss
    
    def train(self, X, y):
        output = self.forward(X)
        backward = self.backward(output, y)


    def forward(self, X):
        output = self.output_layer.forward(self.layer2.forward(self.layer1.forward(X)))
        self.output = output
        return output

    def backward(self, y_pred, y_valid):
        loss = self.loss_categorical_cross_entropy(y_valid, y_pred)
        slope = self.output_layer.activation_function_derivative(y_pred)
