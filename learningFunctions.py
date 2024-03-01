import numpy as np
import activationFunctions as af
import errorFunctions as errfun
import matplotlib.pyplot as plt
from neuralNetwork import NeuralNetwork

from copy import deepcopy


# Funzione utilizzata per copiare parametri da una rete all'altra
def copy_params_in_network(destination_network, source_network):
    for l in range(len(source_network.layers_weights)):
        destination_network.layers_weights[l] = source_network.layers_weights[l].copy()
        destination_network.layers_bias[l] = source_network.layers_bias[l].copy()
    destination_network.hidden_activation_functions = source_network.hidden_activation_functions


# Funzione utilizzata per stampare le caratteritiche della rete
def get_net_structure(network):
    num_hidden_layers = network.number_of_hidden_layers
    input_size = network.layers_weights[0].shape[1]
    output_size = network.layers_weights[num_hidden_layers].shape[0]
    num_neurons_hidden_layers = [network.layers_weights[i].shape[0] for i in range(num_hidden_layers)]
    activation_functions = [network.hidden_activation_functions[i].__name__ for i in range(num_hidden_layers)] + [
        network.hidden_activation_functions[num_hidden_layers].__name__]
    error_function = network.error_function.__name__

    print('num_hidden_layers: ', num_hidden_layers)
    print('input_size: ', input_size)
    print('output_size: ', output_size)
    print('neurons in hidden layers:')
    for neurons in num_neurons_hidden_layers:
        print(neurons)
    print('activation functions:')
    for act_fun in activation_functions:
        print(act_fun)
    print('error_function:', error_function)

    return


# Funzione per ottenere una copia della rete
def duplicate_network(net):
    newNet = deepcopy(net)
    return newNet


# Funzione che effettua la forward propagation
def forward_propagation(network, x):
    # Estrazione dei parametri della rete
    weights = network.layers_weights
    biases = network.layers_bias
    activation_functions = network.hidden_activation_functions
    num_layers = len(network.layers_weights)

    # inizializzazione di z con gli input layer
    z = x
    for l in range(num_layers):
        # Trasformazione lineare tra i pesi e l'input del neurone corrente
        a = np.matmul(weights[l], z) + biases[l]
        z = activation_functions[l](a)

    return z


# Calcolo delle derivate e degli output dei neuroni
def gradients_computation(network, x):
    # Estrazione dei parametri della rete
    weights = network.layers_weights
    biases = network.layers_bias
    activation_functions = network.hidden_activation_functions
    num_layers = len(network.layers_weights)

    result_mul = []
    layer_outputs = []
    activation_derivatives = []

    # inserimento degli input nel vettore layer_outputs
    layer_outputs.append(x)

    for l in range(num_layers):
        # Trasformazione lineare tra i pesi e l'input del neurone corrente
        result_mul.append(np.matmul(weights[l], layer_outputs[l]) + biases[l])

        # Ottenimento della derivata della funzione di attivazione
        z, derivative_activation = activation_functions[l](result_mul[l], 1)
        activation_derivatives.append(derivative_activation)
        layer_outputs.append(z)

    return layer_outputs, activation_derivatives


# Funzione di back-propagation
def back_propagation(network, input_activations, layer_outputs, target, error_function):
    # Estrazione dei parametri della rete
    weights = network.layers_weights
    num_layers = len(network.layers_weights)

    # Calcolo del delta dell'ultimo strato
    output_error_derivative = error_function(layer_outputs[-1], target, 1)
    delta = [input_activations[-1] * output_error_derivative]

    # Calcolo del delta dei livelli precedenti
    for l in range(num_layers - 1, 0, -1):
        error_derivative = input_activations[l - 1] * np.matmul(weights[l].transpose(), delta[0])
        delta.insert(0, error_derivative)

    # Inizializzazione pesi e bias per il calcolo del gradiente
    weight_gradients = []
    bias_gradients = []

    # Calcolo dei gradienti dei pesi e dei bias per ciascuno strato
    for l in range(num_layers):
        # Calcolo del gradiente dei pesi per lo strato corrente
        weight_gradient = np.matmul(delta[l], layer_outputs[l].transpose())
        weight_gradients.append(weight_gradient)

        # Calcolo del gradiente del bias per lo strato corrente
        bias_gradient = np.sum(delta[l], axis=1, keepdims=True)
        bias_gradients.append(bias_gradient)

    return weight_gradients, bias_gradients


# Funzione discesa del gradiente per l'aggiornamento dei pesi
def gradient_descent(network, learning_rate, derivative_weights, derivative_biases):
    num_layers = len(network.layers_weights)
    for l in range(num_layers):
        network.layers_weights[l] = network.layers_weights[l] - learning_rate * derivative_weights[l]
        network.layers_bias[l] = network.layers_bias[l] - learning_rate * derivative_biases[l]

    return network


# Funzione RProp per l'aggiornamento dei pesi per reti multistrato
def rprop_training_phase(network, derW, derB, deltaW, deltaB, oldDerW, oldDerB, posEta=1.2, negEta=0.5, deltaMax=50,
                         deltaMin=0.00001):
    # Aggiornamento dei pesi
    for l in range(len(network.layers_weights)):
        for k in range(len(derW[l])):
            for m in range(len(derW[l][k])):
                # Se la derivata ha lo stesso segno per i pesi di due epoche contigue
                if oldDerW[l][k][m] * derW[l][k][m] > 0:
                    deltaW[l][k][m] = min(deltaW[l][k][m] * posEta, deltaMax)

                elif oldDerW[l][k][m] * derW[l][k][m] < 0:
                    deltaW[l][k][m] = max(deltaW[l][k][m] * negEta, deltaMin)

                # Aggiornamento derivata dei pesi precedenti con quelli correnti
                oldDerW[l][k][m] = derW[l][k][m]

        # Aggiornare i pesi utilizzando il segno delle derivate e le dimensioni dei passi
        network.layers_weights[l] -= np.sign(derW[l]) * deltaW[l]

    # Aggiornamento dei pesi
    for l in range(len(network.layers_bias)):
        for k in range(len(derB[l])):
            # Se la derivata ha lo stesso segno per i bias di due epoche contigue
            if oldDerB[l][k][0] * derB[l][k][0] > 0:
                deltaB[l][k][0] = min(deltaB[l][k][0] * posEta, deltaMax)

            elif oldDerB[l][k][0] * derB[l][k][0] < 0:
                deltaB[l][k][0] = max(deltaB[l][k][0] * negEta, deltaMin)

            # Aggiornamento derivata dei pesi precedenti con quelli correnti
            oldDerB[l][k][0] = derB[l][k][0]

        # Aggiornare i bias utilizzando il segno delle derivate e le dimensioni dei passi
        network.layers_bias[l] -= np.sign(derB[l]) * deltaB[l]

    return network


# Processo di apprendimento per la rete neurale
def train_neural_network(net, X_train, Y_train, X_val=[], Y_val=[], max_epochs=100, learning_rate=0.1):
    training_errors = []
    validation_errors = []
    training_accuracy = []
    validation_accuracy = []
    error_function = net.error_function

    # Inizializzazione delta e derivate precedenti
    delta_weights, delta_biases, old_derivative_weights, old_derivative_biases = None, None, None, None

    # Inizializzazione training
    Y_net_train = forward_propagation(net, X_train)
    train_error = error_function(Y_net_train, Y_train)
    training_errors.append(train_error)

    # Inizializzazione best_net
    Y_net_val = forward_propagation(net, X_val)
    val_error = error_function(Y_net_val, Y_val)
    validation_errors.append(val_error)

    min_val_error = val_error
    best_net = duplicate_network(net)

    accuracy_train = compute_accuracy(Y_net_train, Y_train)
    accuracy_vali = compute_accuracy(Y_net_val, Y_val)
    training_accuracy.append(accuracy_train)
    validation_accuracy.append(accuracy_vali)
    print(f'0/{max_epochs}, Training Accuracy: {accuracy_train}, Validation Accuracy: {accuracy_vali}')

    # Inizio fase di apprendimento
    for epoch in range(max_epochs):
        # Gradient descent e Back-propagation
        layer_z, layer_da = gradients_computation(net, X_train)
        derivative_weights, derivative_biases = back_propagation(net, layer_da, layer_z, Y_train, error_function)

        if (epoch == 0):
            # Aggiornamento pesi tramite discesa del gradiente
            net = gradient_descent(net, learning_rate, derivative_weights, derivative_biases)

            # Inizializzazione dei pesi e dei bias per la funzione RProp
            delta_weights = [[[0.1 for _ in row] for row in sub_list] for sub_list in derivative_weights]
            delta_biases = [[[0.1 for _ in row] for row in sub_list] for sub_list in derivative_biases]

            old_derivative_weights = deepcopy(derivative_weights)
            old_derivative_biases = deepcopy(derivative_biases)
        else:
            # Aggiornamento della rete utilizzando utilizzando la funzione RProp
            net = rprop_training_phase(net, derivative_weights, derivative_biases, delta_weights, delta_biases,
                                       old_derivative_weights, old_derivative_biases)

        # Forward propagation per training set
        Y_net_train = forward_propagation(net, X_train)
        train_error = error_function(Y_net_train, Y_train)
        training_errors.append(train_error)

        # Fase di validation
        Y_net_val = forward_propagation(net, X_val)
        val_error = error_function(Y_net_val, Y_val)
        validation_errors.append(val_error)

        # Trova l'errore minimo e la rete migliore
        if val_error < min_val_error:
            min_val_error = val_error
            best_net = duplicate_network(net)

        accuracy_train = compute_accuracy(Y_net_train, Y_train)
        accuracy_vali = compute_accuracy(Y_net_val, Y_val)
        training_accuracy.append(accuracy_train)
        validation_accuracy.append(accuracy_vali)
        print(f'{epoch + 1}/{max_epochs}, Training Accuracy: {accuracy_train}, Validation Accuracy: {accuracy_vali}',
              end='\r')

    copy_params_in_network(net, best_net)

    return training_errors, validation_errors, training_accuracy, validation_accuracy


# Funzione utilizzata per calcolare l'accuratezza della rete
def compute_accuracy(predictions, targets):
    num_samples = targets.shape[1]

    # Applica la funzione softmax alle previsioni della rete
    softmax_predictions = errfun.softmax(predictions)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne
    predicted_classes = np.argmax(softmax_predictions, axis=0)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne negli obiettivi desiderati
    target_classes = np.argmax(targets, axis=0)

    # Confronta gli indici predetti con gli indici degli obiettivi desiderati e calcola l'accuratezza
    correct_predictions = np.sum(predicted_classes == target_classes)
    accuracy = correct_predictions / num_samples

    return accuracy


# Funzione che permette di calcolare l'accuratezza su input diversi
def network_accuracy(net, X, target):
    y_net = forward_propagation(net, X)
    return compute_accuracy(y_net, target)


# Funzione per ottenere un esempio di predizione della rete
def test_prediction(network, train_mia_net, x, Xtest):
    ix = np.reshape(Xtest[:, x], (28, 28))
    plt.figure()
    plt.imshow(ix, 'gray')
    y_net_trained = forward_propagation(train_mia_net, Xtest[:, x:x + 1])
    y_net = forward_propagation(network, Xtest[:, x:x + 1])
    # Utilizza la funzione softmax per ottenere valori probabilistici
    y_net = errfun.softmax(y_net)
    y_net_trained = errfun.softmax(y_net_trained)
    print('y_net:', y_net)
    print('y_net_trained:', y_net_trained)
