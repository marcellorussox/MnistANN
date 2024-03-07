from uninaannpy import error_functions as errfun

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


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


def get_net_structure(net):
    """
    Ottiene e stampa le caratteristiche della rete neurale.

    Args:
        net (NeuralNetwork): Oggetto della rete neurale da analizzare.
    """
    # Ottiene il numero di strati nascosti
    num_hidden_layers = net.number_of_hidden_layers

    # Ottiene le dimensioni dell'input e dell'output
    input_size = net.layers_weights[0].shape[1]
    output_size = net.layers_weights[num_hidden_layers].shape[0]

    # Ottiene il numero di neuroni per ogni strato nascosto
    num_neurons_hidden_layers = [net.layers_weights[layer].shape[0] for layer in range(num_hidden_layers)]

    # Ottiene le funzioni di attivazione per ogni strato, incluso quello di output
    activation_functions = [net.hidden_activation_functions[layer].__name__ for layer in range(num_hidden_layers)] + [
        net.hidden_activation_functions[num_hidden_layers].__name__]

    # Ottiene il nome della funzione di errore
    error_function = net.error_function.__name__

    # Stampa delle caratteristiche della rete
    print('Numero di strati nascosti:', num_hidden_layers)
    print('Dimensione dell\'input:', input_size)
    print('Dimensione dell\'output:', output_size)
    print('Neuroni negli strati nascosti:', ', '.join(map(str, num_neurons_hidden_layers)))
    print('Funzioni di attivazione:', ', '.join(activation_functions))
    print('Funzione di errore:', error_function)


def duplicate_network(net):
    """
    Crea una copia profonda della rete neurale.

    Args:
        net (NeuralNetwork): Rete neurale da duplicare.

    Returns:
        NeuralNetwork: Copia della rete neurale.
    """
    return deepcopy(net)


def forward_propagation(net, input_data):
    """
    Esegue la propagazione in avanti attraverso la rete neurale.

    Args:
        net (NeuralNetwork): Rete neurale.
        input_data (ndarray): Dati di input.

    Returns:
        ndarray: Uscita della rete neurale dopo la propagazione in avanti.
    """
    # Estrae i parametri della rete
    weights = net.layers_weights
    biases = net.layers_biases
    activation_functions = net.hidden_activation_functions
    num_layers = len(net.layers_weights)

    # Inizializza z con i dati di input
    z = input_data
    for layer in range(num_layers):
        # Calcola l'output del layer corrente
        a = np.matmul(weights[layer], z) + biases[layer]
        z = activation_functions[layer](a)

    return z


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


def back_propagation(network, input_activations, layer_outputs, target, error_function):
    """
    Esegue la back-propagation per calcolare i gradienti dei pesi e dei bias.

    Args:
        network (NeuralNetwork): La rete neurale.
        input_activations (list of numpy.ndarray): Le attivazioni di input di ciascun layer.
        layer_outputs (list of numpy.ndarray): Gli output di ciascun layer.
        target (numpy.ndarray): Il target desiderato per l'output della rete.
        error_function (function): La funzione di errore utilizzata per calcolare il gradiente.

    Returns:
        tuple: Una tupla contenente i gradienti dei pesi e dei bias per ciascuno strato.
    """
    # Estrazione dei parametri della rete
    weights = network.layers_weights
    num_layers = len(network.layers_weights)

    # Inizializzazione dei gradienti dei pesi e dei bias
    weight_gradients = []
    bias_gradients = []

    # Calcolo dei gradienti dei pesi e dei bias per ciascuno strato, partendo dallo strato di output
    for layer in range(num_layers - 1, -1, -1):
        # Calcolo del delta per lo strato corrente
        if layer == num_layers - 1:
            # Calcolo del delta dell'ultimo strato
            output_error_derivative = error_function(layer_outputs[-1], target, 1)
            delta = [input_activations[-1] * output_error_derivative]
        else:
            # Calcolo del delta per gli strati intermedi
            error_derivative = input_activations[layer] * np.matmul(weights[layer + 1].T, delta[0])
            delta = [error_derivative]

        # Calcolo del gradiente dei pesi per lo strato corrente
        weight_gradient = np.matmul(delta[0], layer_outputs[layer].T)
        weight_gradients.insert(0, weight_gradient)

        # Calcolo del gradiente del bias per lo strato corrente
        bias_gradient = np.sum(delta[0], axis=1, keepdims=True)
        bias_gradients.insert(0, bias_gradient)

    return weight_gradients, bias_gradients


def gradient_descent(net, learning_rate, weights_der, biases_der):
    """
    Aggiorna i pesi e i bias della rete neurale utilizzando il gradiente discendente.

    Args:
        net (NeuralNetwork): Rete neurale da aggiornare.
        learning_rate (float): Tasso di apprendimento per controllare la dimensione dei passi dell'aggiornamento.
        weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
        biases_der (list): Lista dei gradienti dei bias per ciascuno strato.

    Returns:
        NeuralNetwork: Rete neurale aggiornata.
    """
    for layer in range(len(net.layers_weights)):
        # Aggiornamento dei pesi utilizzando il gradiente discendente
        net.layers_weights[layer] -= learning_rate * weights_der[layer]

        # Aggiornamento dei bias utilizzando il gradiente discendente
        net.layers_biases[layer] -= learning_rate * biases_der[layer]

    return net


def rprop(network, weights_der, biases_der, weights_delta, biases_delta, weights_der_prev,
          biases_der_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001):
    """
    Funzione RProp per l'aggiornamento dei pesi per reti multistrato.

    Args:
        network (NeuralNetwork): Rete neurale.
        weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
        biases_der (list): Lista dei gradienti dei bias per ciascuno strato.
        weights_delta (list): Lista dei delta dei pesi per ciascuno strato.
        biases_delta (list): Lista dei delta dei bias per ciascuno strato.
        weights_der_prev (list): Lista dei gradienti dei pesi della precedente iterazione.
        biases_der_prev (list): Lista dei gradienti dei bias della precedente iterazione.
        eta_pos (float): Fattore di aggiornamento dei delta per derivata positiva (default: 1.2).
        eta_neg (float): Fattore di aggiornamento dei delta per derivata negativa (default: 0.5).
        delta_max (float): Limite superiore per il delta (default: 50).
        delta_min (float): Limite inferiore per il delta (default: 0.00001).

    Returns:
        NeuralNetwork: Rete neurale aggiornata con il metodo RProp.
    """
    for layer in range(len(network.layers_weights)):
        layer_weights = network.layers_weights[layer]
        layer_biases = network.layers_biases[layer]

        for num_rows in range(len(weights_der[layer])):
            for num_cols in range(len(weights_der[layer][num_rows])):
                # Calcolo della nuova dimensione del delta per i pesi
                if weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][num_cols] > 0:
                    weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] * eta_pos,
                                                                   delta_max)
                elif weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][num_cols] < 0:
                    weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] * eta_neg,
                                                                   delta_min)

                # Aggiornamento dei pesi
                layer_weights[num_rows][num_cols] -= (np.sign(weights_der[layer][num_rows][num_cols]) *
                                                      weights_delta[layer][num_rows][num_cols])

                # Aggiornamento dei gradienti dei pesi precedenti
                weights_der_prev[layer][num_rows][num_cols] = weights_der[layer][num_rows][num_cols]

            # Calcolo della nuova dimensione del delta per i bias
            if biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0] > 0:
                biases_delta[layer][num_rows][0] = min(biases_delta[layer][num_rows][0] * eta_pos, delta_max)
            elif biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0] < 0:
                biases_delta[layer][num_rows][0] = max(biases_delta[layer][num_rows][0] * eta_neg, delta_min)

            # Aggiornamento dei bias
            layer_biases[num_rows][0] -= np.sign(biases_der[layer][num_rows][0]) * biases_delta[layer][num_rows][0]

            # Aggiornamento dei gradienti dei bias precedenti
            biases_der_prev[layer][num_rows][0] = biases_der[layer][num_rows][0]

    return network


def rprop_plus(network, weights_der, biases_der, weights_delta, biases_delta, weights_der_prev,
               biases_der_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001):
    """
    Funzione RProp+ per l'aggiornamento dei pesi per reti multistrato.

    Args:
        network (NeuralNetwork): Rete neurale.
        weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
        biases_der (list): Lista dei gradienti dei bias per ciascuno strato.
        weights_delta (list): Lista dei delta dei pesi per ciascuno strato.
        biases_delta (list): Lista dei delta dei bias per ciascuno strato.
        weights_der_prev (list): Lista dei gradienti dei pesi della precedente iterazione.
        biases_der_prev (list): Lista dei gradienti dei bias della precedente iterazione.
        eta_pos (float): Fattore di aggiornamento dei delta per derivata positiva (default: 1.2).
        eta_neg (float): Fattore di aggiornamento dei delta per derivata negativa (default: 0.5).
        delta_max (float): Limite superiore per il delta (default: 50).
        delta_min (float): Limite inferiore per il delta (default: 0.00001).

    Returns:
        NeuralNetwork: Rete neurale aggiornata con il metodo RProp+.
    """
    for layer in range(len(network.layers_weights)):
        layer_weights = network.layers_weights[layer]
        layer_biases = network.layers_biases[layer]

        # Inizializzazione delle liste dei delta per i pesi e i bias per l'attuale strato
        layer_weights_delta = [[0] * len(row) for row in weights_delta[layer]]
        layer_biases_delta = [[0] * len(row) for row in biases_delta[layer]]

        for num_rows in range(len(weights_der[layer])):
            for num_cols in range(len(weights_der[layer][num_rows])):
                weight_der_product = weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][
                    num_cols]

                if weight_der_product < 0:
                    weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] * eta_neg,
                                                                   delta_min)

                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -layer_weights_delta[num_rows][num_cols]

                    # Aggiornamento della derivata del peso
                    weights_der_prev[layer][num_rows][num_cols] = 0

                # Calcolo della nuova dimensione del delta per i pesi
                elif weight_der_product > 0:
                    weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] * eta_pos,
                                                                   delta_max)

                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -(np.sign(weights_der[layer][num_rows][num_cols]) *
                                                                weights_delta[layer][num_rows][num_cols])
                else:
                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -(np.sign(weights_der[layer][num_rows][num_cols]) *
                                                                weights_delta[layer][num_rows][num_cols])

                # Aggiornamento dei pesi
                layer_weights[num_rows][num_cols] += layer_weights_delta[num_rows][num_cols]

                # Aggiornamento dei gradienti dei bias precedenti
                weights_der_prev[layer][num_rows][num_cols] = weights_der[layer][num_rows][num_cols]

            biases_der_product = biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0]

            # Calcolo della nuova dimensione del delta per i bias
            if biases_der_product < 0:
                biases_delta[layer][num_rows][0] = max(biases_delta[layer][num_rows][0] * eta_neg, delta_min)

                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -layer_biases_delta[num_rows][0]

                # Aggiornamento della derivata del bias
                biases_der[layer][num_rows][0] = 0

            elif biases_der_product > 0:
                biases_delta[layer][num_rows][0] = min(biases_delta[layer][num_rows][0] * eta_pos, delta_max)

                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])

            else:
                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])
            # Aggiornamento dei bias
            layer_biases[num_rows][0] += layer_biases_delta[num_rows][0]

            # Aggiornamento dei gradienti dei bias precedenti
            biases_der_prev[layer][num_rows][0] = biases_der[layer][num_rows][0]

    return network


def irprop(network, weights_der, biases_der, weights_delta, biases_delta, weights_der_prev,
           biases_der_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001):
    """
    Funzione iRProp- per l'aggiornamento dei pesi per reti multistrato.

    Args:
        network (NeuralNetwork): Rete neurale.
        weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
        biases_der (list): Lista dei gradienti dei bias per ciascuno strato.
        weights_delta (list): Lista dei delta dei pesi per ciascuno strato.
        biases_delta (list): Lista dei delta dei bias per ciascuno strato.
        weights_der_prev (list): Lista dei gradienti dei pesi della precedente iterazione.
        biases_der_prev (list): Lista dei gradienti dei bias della precedente iterazione.
        eta_pos (float): Fattore di aggiornamento dei delta per derivata positiva (default: 1.2).
        eta_neg (float): Fattore di aggiornamento dei delta per derivata negativa (default: 0.5).
        delta_max (float): Limite superiore per il delta (default: 50).
        delta_min (float): Limite inferiore per il delta (default: 0.00001).

    Returns:
        NeuralNetwork: Rete neurale aggiornata con il metodo iRprop-.
    """
    for layer in range(len(network.layers_weights)):
        layer_weights = network.layers_weights[layer]
        layer_biases = network.layers_biases[layer]

        for num_rows in range(len(weights_der[layer])):
            for num_cols in range(len(weights_der[layer][num_rows])):
                # Calcolo della nuova dimensione del delta per i pesi
                if weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][num_cols] > 0:
                    weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] * eta_pos,
                                                                   delta_max)
                elif weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][num_cols] < 0:
                    weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] * eta_neg,
                                                                   delta_min)

                    weights_der_prev[layer][num_rows][num_cols] = 0

                # Aggiornamento dei pesi
                layer_weights[num_rows][num_cols] -= (np.sign(weights_der[layer][num_rows][num_cols]) *
                                                      weights_delta[layer][num_rows][num_cols])

                # Aggiornamento dei gradienti dei pesi precedenti
                weights_der_prev[layer][num_rows][num_cols] = weights_der[layer][num_rows][num_cols]

            # Calcolo della nuova dimensione del delta per i bias
            if biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0] > 0:
                biases_delta[layer][num_rows][0] = min(biases_delta[layer][num_rows][0] * eta_pos, delta_max)
            elif biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0] < 0:
                biases_delta[layer][num_rows][0] = max(biases_delta[layer][num_rows][0] * eta_neg, delta_min)

                biases_der_prev[layer][num_rows][0] = 0

            # Aggiornamento dei bias
            layer_biases[num_rows][0] -= np.sign(biases_der[layer][num_rows][0]) * biases_delta[layer][num_rows][0]

            # Aggiornamento dei gradienti dei bias precedenti
            biases_der_prev[layer][num_rows][0] = biases_der[layer][num_rows][0]

    return network


def irprop_plus(network, weights_der, biases_der, weights_delta, biases_delta, weights_der_prev,
                biases_der_prev, train_error, train_error_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001):
    """
    Funzione iRProp+ per l'aggiornamento dei pesi per reti multistrato.

    Args:
        network (NeuralNetwork): Rete neurale.
        weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
        biases_der (list): Lista dei gradienti dei bias per ciascuno strato.
        weights_delta (list): Lista dei delta dei pesi per ciascuno strato.
        biases_delta (list): Lista dei delta dei bias per ciascuno strato.
        weights_der_prev (list): Lista dei gradienti dei pesi della precedente iterazione.
        biases_der_prev (list): Lista dei gradienti dei bias della precedente iterazione.
        eta_pos (float): Fattore di aggiornamento dei delta per derivata positiva (default: 1.2).
        eta_neg (float): Fattore di aggiornamento dei delta per derivata negativa (default: 0.5).
        delta_max (float): Limite superiore per il delta (default: 50).
        delta_min (float): Limite inferiore per il delta (default: 0.00001).

    Returns:
        NeuralNetwork: Rete neurale aggiornata con il metodo iRprop+.
    """
    for layer in range(len(network.layers_weights)):
        layer_weights = network.layers_weights[layer]
        layer_biases = network.layers_biases[layer]

        # Inizializzazione delle liste dei delta per i pesi e i bias per l'attuale strato
        layer_weights_delta = [[0] * len(row) for row in weights_delta[layer]]
        layer_biases_delta = [[0] * len(row) for row in biases_delta[layer]]

        for num_rows in range(len(weights_der[layer])):
            for num_cols in range(len(weights_der[layer][num_rows])):
                weight_der_product = weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][
                    num_cols]

                if weight_der_product < 0:
                    weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] * eta_neg,
                                                                   delta_min)

                    if train_error > train_error_prev:
                        # Aggiornamento del delta del peso
                        layer_weights_delta[num_rows][num_cols] = -layer_weights_delta[num_rows][num_cols]

                    # Aggiornamento della derivata del peso
                    weights_der_prev[layer][num_rows][num_cols] = 0

                # Calcolo della nuova dimensione del delta per i pesi
                elif weight_der_product > 0:
                    weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] * eta_pos,
                                                                   delta_max)

                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -(np.sign(weights_der[layer][num_rows][num_cols]) *
                                                                weights_delta[layer][num_rows][num_cols])
                else:
                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -(np.sign(weights_der[layer][num_rows][num_cols]) *
                                                                weights_delta[layer][num_rows][num_cols])

                # Aggiornamento dei pesi
                layer_weights[num_rows][num_cols] += layer_weights_delta[num_rows][num_cols]

                # Aggiornamento dei gradienti dei bias precedenti
                weights_der_prev[layer][num_rows][num_cols] = weights_der[layer][num_rows][num_cols]

            biases_der_product = biases_der_prev[layer][num_rows][0] * biases_der[layer][num_rows][0]

            # Calcolo della nuova dimensione del delta per i bias
            if biases_der_product < 0:
                biases_delta[layer][num_rows][0] = max(biases_delta[layer][num_rows][0] * eta_neg, delta_min)

                if train_error > train_error_prev:
                    # Aggiornamento del delta del bias
                    layer_biases_delta[num_rows][0] = -layer_biases_delta[num_rows][0]

                # Aggiornamento della derivata del bias
                biases_der[layer][num_rows][0] = 0

            elif biases_der_product > 0:
                biases_delta[layer][num_rows][0] = min(biases_delta[layer][num_rows][0] * eta_pos, delta_max)

                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])

            else:
                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])
            # Aggiornamento dei bias
            layer_biases[num_rows][0] += layer_biases_delta[num_rows][0]

            # Aggiornamento dei gradienti dei bias precedenti
            biases_der_prev[layer][num_rows][0] = biases_der[layer][num_rows][0]

    return network


def train_neural_network(net, train_in, train_labels, validation_in, validation_labels, epochs=100,
                         learning_rate=0.1):
    """
    Processo di apprendimento per la rete neurale.

    Args:
        net (NeuralNetwork): Rete neurale da addestrare.
        train_in (numpy.ndarray): Dati di input per il training.
        train_labels (numpy.ndarray): Target desiderati per i dati di input di training.
        validation_in (numpy.ndarray): Dati di input per la validazione.
        validation_labels (numpy.ndarray): Target desiderati per i dati di input di validazione.
        epochs (int, optional): Numero massimo di epoche per il training. Default: 100.
        learning_rate (float, optional): Tasso di apprendimento per il gradiente discendente. Default: 0.1.

    Returns:
        tuple: Una tupla contenente:
            - train_errors (list): Lista degli errori di training per ogni epoca.
            - validation_errors (list): Lista degli errori di validazione per ogni epoca.
            - train_accuracies (list): Lista delle accuratezze di training per ogni epoca.
            - validation_accuracies (list): Lista delle accuratezze di validazione per ogni epoca.
    """
    train_errors = []
    validation_errors = []
    train_accuracies = []
    validation_accuracies = []
    error_function = net.error_function

    # Inizializzazione delta e derivate precedenti
    weights_delta, biases_delta, weights_der_prev, biases_der_prev = None, None, None, None

    # Inizializzazione training
    train_net_out = forward_propagation(net, train_in)
    train_error = error_function(train_net_out, train_labels)
    train_errors.append(train_error)

    # Inizializzazione best_net
    validation_net_out = forward_propagation(net, validation_in)
    val_error = error_function(validation_net_out, validation_labels)
    validation_errors.append(val_error)

    min_val_error = val_error
    best_net = duplicate_network(net)

    train_accuracy = compute_accuracy(train_net_out, train_labels)
    validation_accuracy = compute_accuracy(validation_net_out, validation_labels)
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)
    print(f'\n0/{epochs}\n'
          f'Training Accuracy: {format_percentage(train_accuracy)}%,\n'
          f'Validation Accuracy: {format_percentage(validation_accuracy)}%\n')

    # Inizio fase di apprendimento
    for epoch in range(epochs):
        # Gradient descent e Back-propagation
        layer_out, layer_act_fun_der = compute_gradients(net, train_in)
        weights_der, biases_der = back_propagation(net, layer_act_fun_der, layer_out, train_labels, error_function)

        train_error_prev = 9999

        if epoch == 0:
            # Aggiornamento pesi tramite discesa del gradiente
            net = gradient_descent(net, learning_rate, weights_der, biases_der)

            # Inizializzazione dei pesi e dei bias per la funzione RProp
            weights_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in weights_der]
            biases_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in biases_der]

            weights_der_prev = deepcopy(weights_der)
            biases_der_prev = deepcopy(biases_der)
        else:
            # Aggiornamento della rete utilizzando la funzione RProp
            net = irprop_plus(net, weights_der, biases_der, weights_delta, biases_delta,
                              weights_der_prev, biases_der_prev, train_error, train_error_prev)

        if epoch > 0:
            train_error_prev = train_error

        # Forward propagation per training set
        train_net_out = forward_propagation(net, train_in)
        train_error = error_function(train_net_out, train_labels)
        train_errors.append(train_error)

        # Fase di validation
        validation_net_out = forward_propagation(net, validation_in)
        val_error = error_function(validation_net_out, validation_labels)
        validation_errors.append(val_error)

        # Trova l'errore minimo e la rete migliore
        if val_error < min_val_error:
            min_val_error = val_error
            best_net = duplicate_network(net)

        train_accuracy = compute_accuracy(train_net_out, train_labels)
        validation_accuracy = compute_accuracy(validation_net_out, validation_labels)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        print(f'\n{epoch + 1}/{epochs}\n'
              f'Training Accuracy: {format_percentage(train_accuracy)}%,\n'
              f'Validation Accuracy: {format_percentage(validation_accuracy)}%\n')

    copy_params_in_network(net, best_net)

    return train_errors, validation_errors, train_accuracies, validation_accuracies


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
    softmax_predictions = errfun.softmax(output)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne
    predicted_classes = np.argmax(softmax_predictions, axis=0)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne negli obiettivi desiderati
    target_classes = np.argmax(labels, axis=0)

    # Confronta gli indici predetti con gli indici degli obiettivi desiderati e calcola l'accuratezza
    correct_predictions = np.sum(predicted_classes == target_classes)
    accuracy = correct_predictions / num_samples

    return accuracy


def network_accuracy(net, input_data, labels):
    """
    Calcola l'accuratezza della rete neurale su un insieme di dati di input e target specificato.

    Args:
        net (NeuralNetwork): Oggetto della rete neurale.
        input_data (numpy.ndarray): Dati di input su cui valutare la rete.
        labels (numpy.ndarray): Target desiderati per i dati di input.

    Returns:
        float: Percentuale di predizioni corrette rispetto ai target desiderati.
    """
    output = forward_propagation(net, input_data)
    return compute_accuracy(output, labels)


def test_prediction(network, train_mia_net, x, Xtest):
    """
    Ottiene un esempio di predizione della rete neurale e lo visualizza insieme all'immagine corrispondente.

    Args:
        network (NeuralNetwork): Rete neurale da testare.
        train_mia_net (NeuralNetwork): Rete neurale addestrata.
        x (int): Indice dell'esempio da testare.
        Xtest (numpy.ndarray): Insieme di dati di test.
    """
    ix = np.reshape(Xtest[:, x], (28, 28))
    plt.figure()
    plt.imshow(ix, 'gray')
    y_net_trained = forward_propagation(train_mia_net, Xtest[:, x:x + 1])
    y_net = forward_propagation(network, Xtest[:, x:x + 1])
    # Utilizza la funzione softmax per ottenere valori probabilistici
    y_net = errfun.softmax(y_net)
    y_net_trained = errfun.softmax(y_net_trained)
    print('Probabilità predette dalla rete non addestrata:')
    for i, probability in enumerate(y_net):
        print(f'Classe {i}: {format_percentage(probability[0])}%')
    print('\nProbabilità predette dalla rete addestrata:')
    for i, probability in enumerate(y_net_trained):
        print(f'Classe {i}: {format_percentage(probability[0])}%')
