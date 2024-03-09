import numpy as np
from matplotlib import pyplot as plt
from uninaannpy import error_functions as ef


def format_percentage(value):
    """
    Formatta un valore o un array di valori come percentuale compresa tra 0 e 100 con solo le prime 5 cifre decimali.

    Args:
        value (float or numpy.ndarray): Valore o array di valori da formattare come percentuale.

    Returns:
        float or numpy.ndarray: Valore o array dei valori percentuali formattati.
    """
    # Se il valore è un array, applica il formattazione a ciascun elemento
    if isinstance(value, np.ndarray):
        return np.round(value * 100, decimals=5)
    # Se il valore è un float, applica il formattazione direttamente
    else:
        return round(value * 100, 5)


def copy_params_in_network(dst_net, src_net):
    """
    Copia i parametri (pesi, bias, funzioni di attivazione) da una rete sorgente a una destinazione.

    Args:
        dst_net (NeuralNetwork): La rete di destinazione.
        src_net (NeuralNetwork): La rete sorgente.
    """
    # Copia dei pesi e dei bias
    for layer in range(len(src_net.layers_weights)):
        dst_net.layers_weights[layer] = src_net.layers_weights[layer].copy()
        dst_net.layers_biases[layer] = src_net.layers_biases[layer].copy()

    # Copia delle funzioni di attivazione
    dst_net.hidden_activation_functions = src_net.hidden_activation_functions


def compute_gradients(net, input_data):
    """
    Calcola le derivate e gli output dei neuroni della rete.

    Args:
        net (NeuralNetwork): La rete neurale.
        input_data (numpy.ndarray): I dati di input.

    Returns:
        tuple: Una tupla contenente gli output dei neuroni di ogni layer e le derivate delle funzioni di attivazione.
    """
    # Estrazione dei parametri della rete
    weights = net.layers_weights
    biases = net.layers_biases
    activation_functions = net.hidden_activation_functions

    num_layers = len(net.layers_weights)

    layer_outputs = [input_data]  # Inizializzazione con l'input

    # Lista per memorizzare le derivate delle funzioni di attivazione
    activation_derivatives = []

    for layer_index in range(num_layers):
        # Trasformazione lineare tra i pesi e l'input del neurone corrente
        result = np.dot(weights[layer_index], layer_outputs[layer_index]) + biases[layer_index]
        layer_output = activation_functions[layer_index](result)  # Output del layer dopo l'attivazione

        # Calcolo della derivata della funzione di attivazione
        derivative_activation = activation_functions[layer_index](result, der=True)[1]

        # Memorizzazione dell'output del layer e della sua derivata di attivazione
        layer_outputs.append(layer_output)
        activation_derivatives.append(derivative_activation)

    return layer_outputs, activation_derivatives


def compute_accuracy(output, labels):
    """
    Calcola l'accuratezza della rete neurale confrontando le previsioni con i target desiderati.

    Args:
        output (numpy.ndarray): Array contenente le previsioni della rete.
        labels (numpy.ndarray): Array contenente i target desiderati.

    Returns:
        float: Percentuale di predizioni corrette rispetto ai target desiderati.
    """
    num_samples = labels.shape[1]

    # Applica la funzione softmax alle previsioni della rete
    softmax_predictions = ef.softmax(output)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne
    predicted_classes = np.argmax(softmax_predictions, axis=0)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne negli obiettivi desiderati
    target_classes = np.argmax(labels, axis=0)

    # Confronta gli indici predetti con gli indici degli obiettivi desiderati e calcola l'accuratezza
    correct_predictions = np.sum(predicted_classes == target_classes)
    accuracy = correct_predictions / num_samples

    return accuracy


def test_prediction(network, trained_net, x, test_in):
    """
    Ottiene un esempio di predizione della rete neurale e lo visualizza insieme all'immagine corrispondente.

    Args:
        network (NeuralNetwork): Rete neurale da testare.
        trained_net (NeuralNetwork): Rete neurale addestrata.
        x (int): Indice dell'esempio da testare.
        test_in (numpy.ndarray): Insieme di dati di test.
    """
    ix = np.reshape(test_in[:, x], (28, 28))
    plt.figure()
    plt.imshow(ix, 'gray')
    trained_net_out = trained_net.forward_propagation(test_in[:, x:x + 1])
    net_out = network.forward_propagation(test_in[:, x:x + 1])
    # Utilizza la funzione softmax per ottenere valori probabilistici
    net_out = ef.softmax(net_out)
    trained_net_out = ef.softmax(trained_net_out)
    print('Probabilità predette dalla rete non addestrata:')
    for i, probability in enumerate(net_out):
        print(f'Classe {i}: {format_percentage(probability[0])}%')
    print('\nProbabilità predette dalla rete addestrata:')
    for i, probability in enumerate(trained_net_out):
        print(f'Classe {i}: {format_percentage(probability[0])}%')
