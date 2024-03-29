import numpy as np


def get_mnist_training(dataset, num_rows, num_cols, validation_index):
    """
    Funzione per creare input e target per il training set a partire da un dataset MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        num_rows (int): Il numero totale di righe nel dataset di training.
        num_cols (int): Il numero totale di colonne nel dataset.
        validation_index (int): L'indice di inizio dei dati di training nel dataset.

    Returns:
        tuple: Una tupla contenente i dati di input di training e le relative etichette.
    """

    # Estrae i dati di training dal dataset, considerando l'indice di validazione
    data_train = dataset[validation_index:num_rows].T

    # Estrae le etichette di training
    train_labels = data_train[0]  # Etichette di training

    # Riduce il numero di etichette a 10
    train_labels = get_mnist_labels(train_labels)

    # Estrae i dati di input di training e normalizza dividendo per 255
    train_input = data_train[1:num_cols]  # Dati di input di training
    train_input = train_input / 255.  # Normalizzazione dei dati divisi per 255

    return train_input, train_labels


def get_mnist_validation(dataset, num_cols, validation_index):
    """
    Crea input e target per il set di validation utilizzando il dataset MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        num_cols (int): Il numero totale di colonne nel dataset.
        validation_index (int): L'indice di fine dei dati di validation nel dataset.

    Returns:
        tuple: Una tupla contenente i dati di input di validation e le relative etichette.
    """

    # Estrae i dati di validation dal dataset
    data_val = dataset[0:validation_index - 1].T

    # Estrae le etichette di validation
    validation_labels = data_val[0]  # Etichette di validation

    # Riduce il numero di etichette a 10
    validation_labels = get_mnist_labels(validation_labels)

    # Estrae i dati di input di validation e normalizza dividendo per 255
    validation_input = data_val[1:num_cols]  # Dati di input di validation
    validation_input = validation_input / 255.  # Normalizzazione dei dati divisi per 255

    return validation_input, validation_labels


def get_mnist_testing(dataset, num_rows, num_cols):
    """
    Crea input e target per il set di testing utilizzando il dataset MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        num_rows (int): Il numero totale di righe nel dataset.
        num_cols (int): Il numero totale di colonne nel dataset.

    Returns:
        tuple: Una tupla contenente i dati di input di testing e le relative etichette.
    """

    # Estrae i dati di testing dal dataset
    data_test = dataset[0:num_rows].T

    # Estrae le etichette di testing
    test_labels = data_test[0]  # Etichette di testing

    # Riduce il numero di etichette a 10
    test_labels = get_mnist_labels(test_labels)

    # Estrae i dati di input di testing
    test_input = data_test[1:num_cols]  # Dati di input di testing

    return test_input, test_labels


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
