import numpy as np


# Funzione di somma dei quadrati
def sumOfSquares(y, t, der=0):
    z = y - t
    if der == 0:
        return 0.5 * np.sum(np.power(z, 2))
    else:
        return z


# Funzione di post-processing softmax
def softmax(y):
    # Per evitare problemi di overflow si sottrae ad y il massimo valore nel vettore y
    y_exp = np.exp(y - y.max(axis=0))
    z = y_exp / np.sum(y_exp, axis=0)
    return z


# Funzione di cross-entropy con softmax
def cross_entropy_softmax(y, t, der=0, epsilon=1e-15):
    softmax_output = softmax(y)

    # Aggiunta di epsilon per evitare log (0)
    softmax_output = np.clip(softmax_output, epsilon, 1 - epsilon)

    if der == 0:
        return -np.sum(t * np.log(softmax_output))
    else:
        return softmax_output - t


# Funzione di cross-entropy
def cross_entropy(y, t, der=0, epsilon=1e-15):
    # Aggiunta di epsilon per evitare log (0)
    y = np.clip(y, epsilon, 1 - epsilon)

    if der == 0:
        return -np.sum(t * np.log(y))
    else:
        return y - t
