from copy import deepcopy
from enum import Enum
import time
import numpy as np

from matplotlib import pyplot as plt
from uninaannpy import error_functions as ef


class RpropType(Enum):
    STANDARD = 1
    RPROP_PLUS = 2
    IRPROP = 3
    IRPROP_PLUS = 4


class NeuralNetwork:
    MU, SIGMA = 0, 0.1

    def __init__(self, hidden_activation_functions, output_activation_function, error_function,
                 input_layer_size, hidden_layers, output_layer_size):
        """
        Costruttore per la creazione della rete neurale.

        Args:
            hidden_activation_functions (list): Lista delle funzioni di attivazione degli strati nascosti.
            output_activation_function (function): Funzione di attivazione dello strato di output.
            error_function (function): Funzione di errore.
            input_layer_size (int): Dimensione dello strato di input.
            hidden_layers (list): Lista che contiene il numero di neuroni per ogni strato nascosto.
            output_layer_size (int): Dimensione dello strato di output.
        """

        self.layers_weights = []
        self.hidden_layers = hidden_layers

        if len(hidden_activation_functions) != len(hidden_layers):
            raise ValueError("Il numero di funzioni di attivazione deve essere uguale al numero di layer!")

        self.hidden_activation_functions = hidden_activation_functions
        hidden_activation_functions.append(output_activation_function)
        self.error_function = error_function
        self.number_of_hidden_layers = len(hidden_layers)
        self.__initialize_parameters(input_layer_size, output_layer_size)

    def __initialize_parameters(self, input_layer_size, output_layer_size):
        """
        Inizializza i pesi e i bias per tutti gli strati della rete neurale.

        Args:
            input_layer_size (int): Dimensione dello strato di input.
            output_layer_size (int): Dimensione dello strato di output.
        """

        hidden_layer_size = self.hidden_layers

        # Inizializzazione dei pesi e del bias per lo strato di input
        self.__initialize_weights(0, hidden_layer_size[0], input_layer_size)

        # Inizializzazione dei pesi e dei bias per gli strati nascosti
        for layer in range(1, self.number_of_hidden_layers):
            self.__initialize_weights(layer, hidden_layer_size[layer], hidden_layer_size[layer - 1])

        # Inizializzazione dei pesi e del bias per lo strato di output
        self.__initialize_weights(self.number_of_hidden_layers, output_layer_size,
                                  hidden_layer_size[self.number_of_hidden_layers - 1])

    def __initialize_weights(self, index, number_of_layer_neurons, input_variables):
        """
        Inizializza i pesi per uno specifico strato della rete neurale.

        Args:
            index (int): Indice dello strato.
            number_of_layer_neurons (int): Numero di neuroni nello strato.
            input_variables (int): Numero di variabili di input.
        """

        self.layers_weights.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons,
                                                                                      input_variables + 1)))

    def get_net_structure(self):
        """
        Ottiene e stampa le caratteristiche della rete neurale.

        Args:
            self (NeuralNetwork): Oggetto della rete neurale da analizzare.
        """
        # Ottiene il numero di strati nascosti
        num_hidden_layers = self.number_of_hidden_layers

        # Ottiene le dimensioni dell'input e dell'output
        input_size = self.layers_weights[0].shape[1]
        output_size = self.layers_weights[num_hidden_layers].shape[0]

        # Ottiene il numero di neuroni per ogni strato nascosto
        num_neurons_hidden_layers = [self.layers_weights[layer].shape[0] for layer in range(num_hidden_layers)]

        # Ottiene le funzioni di attivazione per ogni strato, incluso quello di output
        activation_functions = [self.hidden_activation_functions[layer].__name__ for layer in
                                range(num_hidden_layers)] + [
                                   self.hidden_activation_functions[num_hidden_layers].__name__]

        # Ottiene il nome della funzione di errore
        error_function = self.error_function.__name__

        # Stampa delle caratteristiche della rete
        print('Numero di strati nascosti:', num_hidden_layers)
        print('Dimensione dell\'input:', input_size - 1)
        print('Dimensione dell\'output:', output_size)
        print('Neuroni negli strati nascosti:', ', '.join(map(str, num_neurons_hidden_layers)))
        print('Funzioni di attivazione:', ', '.join(activation_functions))
        print('Funzione di errore:', error_function)

    def duplicate_network(self):
        """
        Crea una copia profonda della rete neurale.

        Args:
            self (NeuralNetwork): Rete neurale da duplicare.

        Returns:
            NeuralNetwork: Copia della rete neurale.
        """
        return deepcopy(self)

    def compute_gradients(self, input_data):
        """
        Calcola le derivate e gli output dei neuroni della rete per la backpropagation.

        Args:
            input_data (numpy.ndarray): I dati di input.

        Returns:
            tuple: Una tupla contenente gli output dei neuroni di ogni layer e le derivate delle funzioni di attivazione.
        """
        # Estrazione dei parametri della rete
        weights = self.layers_weights
        activation_functions = self.hidden_activation_functions

        num_layers = len(self.layers_weights)

        layer_outputs = [input_data]  # Inizializzazione con l'input

        # Lista per memorizzare le derivate delle funzioni di attivazione
        activation_derivatives = []

        for layer in range(num_layers):
            # Trasformazione lineare tra i pesi e l'input del neurone corrente
            result = np.dot(weights[layer][:, 1:], layer_outputs[layer]) + weights[layer][:, 0:1]
            layer_output = activation_functions[layer](result)  # Output del layer dopo l'attivazione

            # Calcolo della derivata della funzione di attivazione
            derivative_activation = activation_functions[layer](result, der=True)

            # Memorizzazione dell'output del layer e della sua derivata di attivazione
            layer_outputs.append(layer_output)
            activation_derivatives.append(derivative_activation)

        return layer_outputs, activation_derivatives

    def forward_propagation(self, input_data):
        """
        Esegue la propagazione in avanti attraverso i pesi e i bias della rete neurale.

        Args:
            input_data (ndarray): Dati di input.

        Returns:
            ndarray: Uscita della rete neurale dopo la propagazione in avanti.
        """
        # Estrae i parametri della rete
        weights = self.layers_weights
        activation_functions = self.hidden_activation_functions
        num_layers = len(self.layers_weights)

        # Creiamo un vettore di 1 di dimensioni num_cols di input_data
        ones_row = np.ones(input_data.shape[1])

        # Inizializza z con i dati di input
        z = input_data
        for layer in range(num_layers):
            # Aggiungiamo la riga di 1 per l'input x_0 il cui peso sarà il bias
            z = np.insert(z, 0, ones_row, axis=0)

            # Calcola l'output del layer corrente
            a = np.matmul(weights[layer], z)
            z = activation_functions[layer](a)

        return z

    def gradient_descent(self, learning_rate, weights_der):
        """
        Aggiorna i pesi e i bias della rete neurale utilizzando la discesa del gradiente.

        Args:
            learning_rate (float): Tasso di apprendimento per controllare la dimensione dei passi dell'aggiornamento.
            weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.

        Returns:
            NeuralNetwork: Rete neurale aggiornata.
        """
        for layer in range(len(self.layers_weights)):
            # Aggiornamento dei pesi utilizzando il gradiente discendente
            self.layers_weights[layer] -= learning_rate * weights_der[layer]

        return self

    def back_propagation(self, input_activations, layer_outputs, target, error_function):
        """
        Esegue la back-propagation per calcolare i gradienti dei pesi e dei bias.

        Args:
            input_activations (list of numpy.ndarray): Le attivazioni di input di ciascun layer.
            layer_outputs (list of numpy.ndarray): Gli output di ciascun layer.
            target (numpy.ndarray): Il target desiderato per l'output della rete.
            error_function (function): La funzione di errore utilizzata per calcolare il gradiente.

        Returns:
            tuple: Una tupla contenente i gradienti dei pesi e dei bias per ciascuno strato.
        """
        # Estrazione dei parametri della rete
        weights = self.layers_weights
        num_layers = len(self.layers_weights)

        # Inizializzazione dei gradienti dei pesi e dei bias
        weight_gradients = []

        # Calcolo dei gradienti dei pesi e dei bias per ciascuno strato, partendo dallo strato di output
        for layer in range(num_layers - 1, -1, -1):
            # Calcolo del delta per lo strato corrente
            if layer == num_layers - 1:
                # Calcolo del delta dell'ultimo strato
                output_error_derivative = error_function(layer_outputs[-1], target, der=True)
                delta = [input_activations[-1] * output_error_derivative]
            else:
                # Calcolo del delta per gli strati intermedi
                error_derivative = input_activations[layer] * np.matmul(weights[layer + 1][:, 1:].T, delta[0])
                delta = [error_derivative]

            # Calcolo del gradiente dei pesi per lo strato corrente
            weight_gradient = np.matmul(delta[0], layer_outputs[layer].T)

            # Calcolo del gradiente del bias per lo strato corrente
            bias_gradient = np.sum(delta[0], axis=1, keepdims=True)
            weight_gradient = np.hstack((bias_gradient, weight_gradient))

            weight_gradients.insert(0, weight_gradient)

        return weight_gradients

    def network_accuracy(self, input_data, labels):
        """
        Calcola l'accuratezza della rete neurale su un insieme di dati di input e target specificato.

        Args:
            input_data (numpy.ndarray): Dati di input su cui valutare la rete.
            labels (numpy.ndarray): Target desiderati per i dati di input.

        Returns:
            float: Percentuale di predizioni corrette rispetto ai target desiderati.
        """
        output = self.forward_propagation(input_data)
        return compute_accuracy(output, labels)

    def print_accuracies(self, title, test_in, test_labels, train_in, train_labels):
        """
        Stampa le accuratezze della rete neurale sui set di test e di addestramento.

        Argomenti:
        - title (str): Il titolo da stampare prima di visualizzare le accuratezze.
        - test_in (numpy.ndarray): Il set di input di test.
        - test_labels (numpy.ndarray): Le etichette di test corrispondenti.
        - train_in (numpy.ndarray): Il set di input di addestramento.
        - train_labels (numpy.ndarray): Le etichette di addestramento corrispondenti.

        Restituisce:
        - net_accuracy_test (float): L'accuratezza della rete neurale sul set di test.

        Stampa il titolo specificato, seguito dall'accuratezza della rete neurale sui set di test e di addestramento.
        """
        print(title)
        net_accuracy_test = self.network_accuracy(test_in, test_labels)
        print(f'Test accuracy: {np.round(net_accuracy_test, 5)}')
        net_accuracy_training = self.network_accuracy(train_in, train_labels)
        print(f'Train accuracy: {np.round(net_accuracy_training, 5)}')
        return net_accuracy_test

    def rprops(self, weights_der, weights_delta, weights_der_prev, layer_weights_difference_prev, train_error,
               train_error_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001,
               rprop_type=RpropType.STANDARD):
        """
        Funzione Rprop per l'aggiornamento dei pesi per reti multistrato. Implementa la versione standard e le tre varianti
        contenute nell'articolo "Empirical evaluation of the improved Rprop learning algorithms". Le varianti vengono
        implementate tramite l'attributo rprop_type.

        Args:
            weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
            weights_delta (list): Lista dei delta dei pesi per ciascuno strato.
            weights_der_prev (list): Lista dei gradienti dei pesi della precedente iterazione.
            layer_weights_difference_prev (list): Lista delle differenze dei pesi della precedente iterazione.
            train_error (float): Errore dell'epoca corrente.
            train_error_prev (float): Errore dell'epoca precedente.
            eta_pos (float): Fattore di aggiornamento dei delta per derivata positiva (default: 1.2).
            eta_neg (float): Fattore di aggiornamento dei delta per derivata negativa (default: 0.5).
            delta_max (float): Limite superiore per il delta (default: 50).
            delta_min (float): Limite inferiore per il delta (default: 0.00001).
            rprop_type (RpropType): Tipo di Rprop da utilizzare (default: RpropType.STANDARD).

        Returns:
            NeuralNetwork: Rete neurale aggiornata con il metodo Rprop.
        """

        # Inizializzazione delle liste dei delta per i pesi e i bias per l'attuale strato
        layer_weights_difference = layer_weights_difference_prev

        for layer in range(len(self.layers_weights)):
            layer_weights = self.layers_weights[layer]

            for num_rows in range(len(weights_der[layer])):
                for num_cols in range(len(weights_der[layer][num_rows])):
                    weight_der_product = weights_der_prev[layer][num_rows][num_cols] * weights_der[layer][num_rows][
                        num_cols]

                    if weight_der_product > 0:
                        # Calcolo della nuova dimensione del delta per i pesi
                        weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] *
                                                                       eta_pos, delta_max)

                        # Aggiornamento della differenza del peso
                        layer_weights_difference[layer][num_rows][num_cols] = -(
                                np.sign(weights_der[layer][num_rows][num_cols])
                                * weights_delta[layer][num_rows][num_cols])

                    elif weight_der_product < 0:
                        # Calcolo della nuova dimensione del delta per i pesi
                        weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] *
                                                                       eta_neg, delta_min)

                        if rprop_type == RpropType.STANDARD or rprop_type == RpropType.IRPROP:
                            # Aggiornamento della differenza del peso
                            layer_weights_difference[layer][num_rows][num_cols] = -(
                                    np.sign(weights_der[layer][num_rows][
                                                num_cols]) *
                                    weights_delta[layer][num_rows][num_cols])
                        else:
                            if rprop_type == RpropType.RPROP_PLUS or train_error > train_error_prev:
                                # Aggiornamento della differenza del peso
                                layer_weights_difference[layer][num_rows][num_cols] = -layer_weights_difference_prev[
                                    layer][num_rows][num_cols]
                            else:
                                # Aggiornamento della differenza del peso
                                layer_weights_difference[layer][num_rows][num_cols] = 0

                        if rprop_type != RpropType.STANDARD:
                            # Aggiornamento della derivata del peso
                            weights_der[layer][num_rows][num_cols] = 0

                    else:
                        # Aggiornamento della differenza del peso
                        layer_weights_difference[layer][num_rows][num_cols] = -(
                                np.sign(weights_der[layer][num_rows][num_cols])
                                * weights_delta[layer][num_rows][num_cols])

                    # Aggiornamento del peso
                    layer_weights[num_rows][num_cols] += layer_weights_difference[layer][num_rows][num_cols]

                    # Aggiornamento del gradiente del peso precedente
                    weights_der_prev[layer][num_rows][num_cols] = weights_der[layer][num_rows][num_cols]

                    layer_weights_difference_prev[layer][num_rows][num_cols] = layer_weights_difference[layer][num_rows][num_cols]

        return layer_weights_difference

    def train_neural_network(self, train_in, train_labels, validation_in, validation_labels, epochs=100,
                             learning_rate=0.00001, rprop_type=RpropType.STANDARD):
        """
        Processo di apprendimento per la rete neurale.

        Args:
            train_in (numpy.ndarray): Dati di input per il training.
            train_labels (numpy.ndarray): Target desiderati per i dati di input di training.
            validation_in (numpy.ndarray): Dati di input per la validazione.
            validation_labels (numpy.ndarray): Target desiderati per i dati di input di validazione.
            epochs (int, optional): Numero massimo di epoche per il training (default: 100).
            learning_rate (float, optional): Tasso di apprendimento per il gradiente discendente (default: 0.1).
            rprop_type (RpropType): Tipo di Rprop da utilizzare (default: RpropType.STANDARD).

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
        error_function = self.error_function

        # Inizializzazione delta e derivate precedenti
        weights_delta, weights_der_prev, layer_weights_difference = None, None, None

        validation_error_prev = 999999
        min_validation_error = 999999
        best_net = self.duplicate_network()

        # Avvia il timer
        start_time = time.time()

        # Inizio fase di apprendimento
        for epoch in range(epochs + 1):

            # Forward propagation per training set
            train_net_out = self.forward_propagation(train_in)
            train_error = error_function(train_net_out, train_labels)
            train_errors.append(train_error)

            # Forward propagation per validation set
            validation_net_out = self.forward_propagation(validation_in)
            validation_error = error_function(validation_net_out, validation_labels)
            validation_errors.append(validation_error)

            train_accuracy = compute_accuracy(train_net_out, train_labels)
            validation_accuracy = compute_accuracy(validation_net_out, validation_labels)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            print(f'\nEpoca: {epoch}/{epochs}   Rprop utilizzata: {rprop_type}\n'
                  f'    Training Accuracy: {np.round(train_accuracy, 5)},       Training Loss: {np.round(train_error, 5)};\n'
                  f'    Validation Accuracy: {np.round(validation_accuracy, 5)},     Validation Loss: {np.round(validation_error, 5)}\n')

            if epoch == epochs:
                break

            # Calcolo gradienti dei pesi e Back-propagation
            layer_out, layer_act_fun_der = self.compute_gradients(train_in)
            weights_der = self.back_propagation(layer_act_fun_der, layer_out, train_labels, error_function)

            if epoch == 0:  # Prima epoca
                # Aggiornamento pesi tramite discesa del gradiente
                self.gradient_descent(learning_rate, weights_der)

                # Inizializzazione dei pesi e dei bias per la funzione Rprop
                weights_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in weights_der]
                layer_weights_difference = [[[0. for _ in row] for row in sub_list] for sub_list in weights_der]

                weights_der_prev = deepcopy(weights_der)
            else:
                # Aggiornamento della rete utilizzando la funzione Rprop
                layer_weights_difference = self.rprops(weights_der, weights_delta, weights_der_prev,
                                                       layer_weights_difference, validation_error,
                                                       validation_error_prev, rprop_type=rprop_type)

            validation_error_prev = validation_error

            # Trova l'errore minimo e la rete migliore
            if validation_error < min_validation_error:
                min_validation_error = validation_error
                best_net = self.duplicate_network()

        # Ferma il timer
        end_time = time.time()

        time_diff = end_time - start_time
        print("L'addestramento ha impiegato", round(time_diff, 5), "secondi per eseguire.")

        best_net.copy_params_in_network(self)

        return train_errors, validation_errors, train_accuracies, validation_accuracies, time_diff

    def copy_params_in_network(self, destination_net):
        """
        Copia i parametri (pesi, funzioni di attivazione) da una rete sorgente a una destinazione.

        Args:
            self (NeuralNetwork): La rete sorgente.
            destination_net (NeuralNetwork): La rete di destinazione.
        """
        # Copia dei pesi e dei bias
        for layer in range(len(self.layers_weights)):
            destination_net.layers_weights[layer] = self.layers_weights[layer].copy()

        # Copia delle funzioni di attivazione
        destination_net.hidden_activation_functions = self.hidden_activation_functions

    def test_prediction(self, x, data_in):
        """
        Ottiene un esempio di predizione della rete neurale e lo visualizza insieme all'immagine corrispondente.

        Args:
            self (NeuralNetwork): Rete neurale da testare.
            x (int): Indice dell'esempio da testare.
            data_in (numpy.ndarray): Insieme di dati di test.
        """
        image = np.reshape(data_in[:, x], (28, 28))
        plt.figure()
        plt.imshow(image, 'gray')
        net_out = self.forward_propagation(data_in[:, x:x + 1])

        # Utilizza la funzione softmax per ottenere valori probabilistici
        net_out = ef.softmax(net_out)
        print('Probabilità predette dalla rete:')
        for i, probability in enumerate(net_out):
            print(f'Classe {i}: {probability[0]}')


def compute_accuracy(output, labels):
    """
    Calcola l'accuratezza della rete neurale confrontando le previsioni con i target desiderati.

    Args:
        output (numpy.ndarray): Array contenente le previsioni della rete.
        labels (numpy.ndarray): Array contenente i target desiderati.

    Returns:
        float: Rapporto tra predizioni corrette e target desiderati.
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
