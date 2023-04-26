from neural_network import Neural_Network
from layer_dense import Layer_Dense
from activation_layer import ActivationLayer
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Define activation functions and their derivatives
def ReLU(inputs):
     '''Activation ReLU'''
     output = np.maximum(0, inputs)
     return output

def ReLU_derivative( inputs):
     '''Derivative of ReLU'''
     output = np.where(inputs > 0, 1, 0)
     return output

def sigmoid(inputs):
     '''Activation sigmoid'''
     output = 1 / (1 + np.exp(-inputs))
     return output

def sigmoid_derivative(inputs):
     '''Derivative of sigmoid'''
     output = sigmoid(inputs) * (1 - sigmoid(inputs))
     return output

def softmax(inputs):
     '''Softmax activation function'''
     exp_values = np.exp(inputs)
     probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
     
     return probabilities

def softmax_derivative(inputs):
     '''Derivative of Softmax'''
     return inputs * (1 - inputs)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
     return 1 - x ** 2

nn = Neural_Network()
nn.add_layer(Layer_Dense(28*28, 100))
nn.add_layer(ActivationLayer(tanh, tanh_derivative))
nn.add_layer(Layer_Dense(100, 100))
nn.add_layer(ActivationLayer(tanh, tanh_derivative))
nn.add_layer(Layer_Dense(100, 10))
nn.add_layer(ActivationLayer(softmax, softmax_derivative))

BATCH_SIZE = 128
X_train = (X_train - X_train.mean()) / X_train.std()
y_train = to_categorical(y_train)
for i in range(1000):
     X = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
     y = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
     nn.train(X, y)
y_pred = nn.predict(X_test)

print(np.sum(np.equal(y_test,np.argmax(y_pred, axis=1))) / len(y_test))
