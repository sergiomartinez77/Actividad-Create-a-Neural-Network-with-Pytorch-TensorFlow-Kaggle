import numpy as np


# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Función de entrenamiento

def entrenar_perceptron(X, y, lr=0.1, epocas=20):

    pesos = np.zeros(X.shape[1])
    bias = 0

    for epoca in range(epocas):

        for i in range(len(X)):

            entrada = X[i]
            salida_esperada = y[i]

            # suma ponderada
            suma = np.dot(entrada, pesos) + bias

            # activación
            salida = sigmoid(suma)

            # convertir a clase
            salida_clase = 1 if salida >= 0.5 else 0

            # error
            error = salida_esperada - salida_clase

            # actualizar pesos
            pesos += lr * error * entrada
            bias += lr * error

        print(f"Época {epoca+1}: Pesos = {pesos}, Bias = {bias}")

    return pesos, bias


#
# Función para probar 

def probar_modelo(X, pesos, bias):

    print("\nPruebas del perceptrón entrenado:")

    for entrada in X:

        suma = np.dot(entrada, pesos) + bias
        salida = sigmoid(suma)

        salida_clase = 1 if salida >= 0.5 else 0

        print(f"Entrada: {entrada} -> Salida: {salida_clase}")


# perceptron AND 

print("\n===== COMPUERTA AND =====")

X_and = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_and = np.array([0,0,0,1])

pesos, bias = entrenar_perceptron(X_and, y_and)

probar_modelo(X_and, pesos, bias)


#perceptron OR

print("\n===== COMPUERTA OR =====")

X_or = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_or = np.array([0,1,1,1])

pesos, bias = entrenar_perceptron(X_or, y_or)

probar_modelo(X_or, pesos, bias)


#Clasificado 

print("\n===== CLASIFICADOR CON 3 ENTRADAS =====")

X_clasificador = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [1,1,1]
])

y_clasificador = np.array([0,0,1,1])

pesos, bias = entrenar_perceptron(X_clasificador, y_clasificador)

probar_modelo(X_clasificador, pesos, bias)