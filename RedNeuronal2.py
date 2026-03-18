import numpy as np
import matplotlib.pyplot as plt


print(" RED NEURONAL MULTICAPA ")


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(0)

hidden_neurons = 4

W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

lr = 0.1
epochs = 10000

losses = []

for epoch in range(epochs):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    error = y - A2
    loss = np.mean(error**2)
    losses.append(loss)

    dA2 = error * sigmoid_deriv(Z2)

    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    dA1 = np.dot(dA2, W2.T) * relu_deriv(Z1)

    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    W2 += lr * dW2
    b2 += lr * db2

    W1 += lr * dW1
    b1 += lr * db1

print("\nResultados XOR:")

for i in range(len(X)):

    Z1 = np.dot(X[i], W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    print(X[i], "->", round(A2[0][0], 4))

plt.figure()
plt.plot(losses)
plt.title("Error vs Epocas (XOR)")
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.show()

print("\nPrueba con nuevos datos")

nuevo = np.array([[1,0]])

Z1 = np.dot(nuevo, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

print("Entrada:", nuevo)
print("Prediccion:", A2)