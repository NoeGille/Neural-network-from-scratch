from neural_network import Neural_Network
from layer_dense import Layer_Dense
from activation_layer import ActivationLayer
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
X_train = (X_train - X_train.mean()) / X_train.std()
y_train = to_categorical(y_train)
X_test = (X_test - X_test.mean()) / X_test.std()
y_test = to_categorical(y_test)


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

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_derivative(z):
    return softmax(z) * (1 - softmax(z))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
     return 1 - np.tanh(x) ** 2

nn = Neural_Network()
nn.add_layer(Layer_Dense(28*28, 100))
nn.add_layer(ActivationLayer(tanh, tanh_derivative))
nn.add_layer(Layer_Dense(100, 100))
nn.add_layer(ActivationLayer(tanh, tanh_derivative))
nn.add_layer(Layer_Dense(100, 10))
nn.add_layer(ActivationLayer(softmax, lambda x:x))

BATCH_SIZE = 16

for i in range(5):
     X = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
     y = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
     nn.train(X, y)
y_pred = nn.predict(X_test)
print(y_pred.argmax(axis=1))
print(y_test.argmax(axis=1))
print(np.sum(np.equal(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))) / len(y_test))



