import numpy as np


def sum_of_squares(net_output, labels, der=False):
    """
    Calcola la funzione di somma dei quadrati o la sua derivata.

    Args:
        net_output (numpy.ndarray): Vettore delle previsioni del modello.
        labels (numpy.ndarray): Vettore dei valori target.
        der (bool, optional): Se è False, calcola la funzione di somma dei quadrati, altrimenti calcola la derivata. Default è False.

    Returns:
        float: Se der=False, restituisce il valore della funzione di somma dei quadrati.
        numpy.ndarray: Se der=True, restituisce il vettore differenza tra net_output e labels.
    """
    z = net_output - labels
    if not der:
        return 0.5 * np.sum(np.power(z, 2))
    else:
        return z


def softmax(net_output):
    """
    Calcola la funzione softmax per ogni elemento del vettore net_output.

    Args:
        y (numpy.ndarray): Vettore di valori.

    Returns:
        numpy.ndarray: Vettore di probabilità normalizzato tramite la funzione softmax.
    """
    # Calcola l'esponenziale di ogni elemento nel vettore y
    y_exp = np.exp(net_output - net_output.max(axis=0))

    # Calcola la somma lungo l'asse 0 del vettore y_exp
    denominator = np.sum(y_exp, axis=0)

    # Calcola la funzione softmax normalizzata
    z = y_exp / denominator

    return z


def cross_entropy_softmax(net_output, labels, der=False, epsilon=1e-15):
    """
    Calcola la cross-entropy tra i valori previsti net_output e i valori target labels utilizzando la funzione softmax.

    Args:
        net_output (numpy.ndarray): Array contenente i valori previsti dalla rete.
        labels (numpy.ndarray): Array contenente i valori target.
        der (bool, optional): Indica se calcolare la derivata della cross-entropy. Default è False.
        epsilon (float, optional): Valore utilizzato per evitare il log(0). Default è 1e-15.

    Returns:
        numpy.ndarray: La cross-entropy tra net_output e labels se der è False, altrimenti la sua derivata.
    """
    softmax_output = softmax(net_output)

    # Aggiunta di epsilon per evitare log(0)
    softmax_output = np.clip(softmax_output, epsilon, 1 - epsilon)

    if not der:
        return -np.sum(labels * np.log(softmax_output))
    else:
        return softmax_output - labels


def cross_entropy(net_output, labels, der=False, epsilon=1e-15):
    """
    Calcola la cross-entropy tra l'output della rete e le etichette fornite.

    Args:
        net_output (numpy.ndarray): Array contenente l'output della rete neurale.
        labels (numpy.ndarray): Array contenente le etichette di classe corrispondenti.
        der (bool, optional): Se True, calcola la derivata della cross-entropy rispetto all'output. Default è False.
        epsilon (float, optional): Valore utilizzato per evitare il logaritmo di zero. Default è 1e-15.

    Returns:
        numpy.ndarray: La cross-entropy tra l'output della rete e le etichette fornite se der è False, altrimenti la sua derivata.
    """
    # Aggiunta di epsilon per evitare log(0)
    net_output = np.clip(net_output, epsilon, 1 - epsilon)

    if not der:
        return -np.sum(labels * np.log(net_output))
    else:
        return net_output - labels
