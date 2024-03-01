import numpy as np


# Funzione identità
def identity(x, der=0):
    if der == 0:
        return x
    # Per il calcolo della derivata della funzione identità
    else:
        return x, 1


# Funzione tangente iperbolica
def tanh(x, der=0):
    y = np.tanh(x)
    if der == 0:
        return y
    # Per il calcolo della derivata della tangente iperbolica
    else:
        return y, 1 - y * y


# Funzione ReLU (Rectified Linear Unit)
def relu(x, der=0):
    if der == 0:
        return np.maximum(0, x)
    # Per il calcolo della derivata della funzione ReLU
    else:
        return np.maximum(0, x), np.where(x > 0, 1, 0)


# Funzione Leaky ReLU (Leaky Rectified Linear Unit)
def leaky_relu(x, der=0, alpha=0.01):
    if der == 0:
        return np.where(x > 0, x, alpha * x)
    # Per il calcolo della derivata della funzione Leaky ReLU
    else:
        return np.where(x > 0, x, alpha * x), np.where(x > 0, 1, alpha)
