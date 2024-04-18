import numpy as np


def get_mnist_training(dataset, validation_index):
    """
    Funzione per creare input e target per il train set a partire dal MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        validation_index (int): L'indice di inizio dei dati di training nel dataset.

    Returns:
        tuple: Una tupla contenente i dati di input di training e le relative etichette.
    """

    # Estrae i dati di training dal dataset, considerando l'indice di validazione
    data_train = dataset[validation_index:].T

    # Estrae le etichette di training
    train_Y = data_train[0]  # Etichette di training

    # Riduce il numero di etichette a 10
    train_Y = get_mnist_labels(train_Y)

    # Estrae i dati di input di training e normalizza dividendo per 255
    train_X = data_train[1:]  # Dati di input di training
    train_X = train_X / 255.  # Normalizzazione dei dati divisi per 255

    return train_X, train_Y


def get_mnist_validation(dataset, validation_index):
    """
    Crea input e target per il set di validation utilizzando il dataset MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        validation_index (int): L'indice di fine dei dati di validation nel dataset.

    Returns:
        tuple: Una tupla contenente i dati di input di validation e le relative etichette.
    """

    # Estrae i dati di validation dal dataset
    data_val = dataset[:validation_index - 1].T

    # Estrae le etichette di validation
    validation_Y = data_val[0]  # Etichette di validation

    # Riduce il numero di etichette a 10
    validation_Y = get_mnist_labels(validation_Y)

    # Estrae i dati di input di validation e normalizza dividendo per 255
    validation_X = data_val[1:]  # Dati di input di validation
    validation_X = validation_X / 255.  # Normalizzazione dei dati divisi per 255

    return validation_X, validation_Y


def get_mnist_test(dataset):
    """
    Crea input e target per il set di test utilizzando il dataset MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.

    Returns:
        tuple: Una tupla contenente i dati di input di test e le relative etichette.
    """

    # Estrae i dati di test dal dataset
    data_test = dataset.T

    # Estrae le etichette di test
    test_Y = data_test[0]  # Etichette di test

    # Riduce il numero di etichette a 10
    test_Y = get_mnist_labels(test_Y)

    # Estrae i dati di input di test
    test_X = data_test[1:]  # Dati di input di test

    return test_X, test_Y


def get_mnist_labels(labels):
    """
    Converte le etichette in formato one-hot.

    Args:
        labels (numpy.ndarray): Array contenente le etichette.

    Returns:
        numpy.ndarray: Array contenente le etichette in formato one-hot.
    """

    labels = np.array(labels)
    num_labels = labels.shape[0]
    num_classes = 10  # Numero di classi (etichette) nel dataset MNIST
    one_hot_labels = np.zeros((num_classes, num_labels), dtype=int)

    for n in range(num_labels):
        label = labels[n]
        one_hot_labels[label][n] = 1

    return one_hot_labels
