from copy import deepcopy

from uninaannpy import activation_functions as af
from uninaannpy import learning_functions as lf
from uninaannpy import utility as ut

import numpy as np

from uninaannpy.learning_functions import RPropType


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

        self.layers_biases = []
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

        # Inizializzazione dei pesi e dei bias per lo strato di input
        self.__initialize_weights(0, hidden_layer_size[0], input_layer_size)
        self.__initialize_bias(0, hidden_layer_size[0])

        # Inizializzazione dei pesi e dei bias per gli strati nascosti
        for i in range(1, self.number_of_hidden_layers):
            self.__initialize_weights(i, hidden_layer_size[i], hidden_layer_size[i])
            self.__initialize_bias(i, hidden_layer_size[i])

        # Inizializzazione dei pesi e dei bias per lo strato di output
        self.__initialize_weights(self.number_of_hidden_layers, output_layer_size,
                                  hidden_layer_size[self.number_of_hidden_layers - 1])
        self.__initialize_bias(self.number_of_hidden_layers, output_layer_size)

    def __initialize_weights(self, index, number_of_layer_neurons, input_variables):
        """
        Inizializza i pesi per uno specifico strato della rete neurale.

        Args:
            index (int): Indice dello strato.
            number_of_layer_neurons (int): Numero di neuroni nello strato.
            input_variables (int): Numero di variabili di input.
        """

        self.layers_weights.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons,
                                                                                      input_variables)))

    def __initialize_bias(self, index, number_of_layer_neurons):
        """
        Inizializza i bias per uno specifico strato della rete neurale.

        Args:
            index (int): Indice dello strato.
            number_of_layer_neurons (int): Numero di neuroni nello strato.
        """

        self.layers_biases.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons, 1)))

    def get_weights(self, layer=0):
        """
        Restituisce i pesi per uno specifico strato della rete neurale.

        Args:
            layer (int, optional): Indice dello strato.

        Returns:
            numpy.ndarray: Pesi per lo strato specificato.
        """

        weights = self.layers_weights
        if layer > 0:
            return weights[layer - 1]
        else:
            return weights

    def get_biases(self, layer=0):
        """
        Restituisce i bias per uno specifico strato della rete neurale.

        Args:
            layer (int, optional): Indice dello strato.

        Returns:
            numpy.ndarray: Bias per lo strato specificato.
        """

        biases = self.layers_biases
        if layer > 0:
            return biases[layer - 1]
        else:
            return biases

    # Funzione per ottenere le funzioni di attivazione
    def get_activation_functions(self, layer=0):
        """
        Restituisce le funzioni di attivazione per uno specifico strato della rete.

        Args:
            layer (int, optional): Indice dello strato.

        Returns:
            list: Funzioni di attivazione del layer specificato.
        """
        if layer > 0:
            return self.hidden_activation_functions[layer - 1]
        else:
            return self.hidden_activation_functions

    # Metodo per modificare le funzioni di attivazione
    def set_activation_function(self, layer_indices, activation_function=af.tanh, layer_type=1):
        """
        Imposta le funzioni di attivazione per la rete neurale.

        Args:
            network (NeuralNetwork): Rete neurale.
            layer_indices (list): Indici degli strati da modificare.
            activation_function (function, optional): Funzione di attivazione da impostare.
            layer_type (int, optional): Tipo di modifica da applicare.
        Returns:
            NeuralNetwork: Rete neurale modificata.
        """
        if layer_type == 0:  # Modifica la funzione di attivazione per gli strati specificati
            if np.isscalar(layer_indices):
                self.hidden_activation_functions[layer_indices - 1] = activation_function
            else:
                for layer, index in enumerate(layer_indices):
                    self.hidden_activation_functions[index - 1] = activation_function[layer]
        elif layer_type == 1:  # Modifica la funzione di attivazione per tutti gli strati interni
            for layer in range(self.number_of_hidden_layers):
                self.hidden_activation_functions[layer] = activation_function
        else:  # Modifica la funzione di attivazione per lo strato di output
            self.hidden_activation_functions[self.number_of_hidden_layers] = activation_function

        # Controllo per verificare se il numero di funzioni di attivazione è uguale al numero di strati interni
        if len(self.hidden_activation_functions) != self.number_of_hidden_layers + 1:
            raise ValueError("Il numero di funzioni di attivazione deve essere uguale al numero di layer!")

        return self

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
        print('Dimensione dell\'input:', input_size)
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

    def forward_propagation(self, input_data):
        """
        Esegue la propagazione in avanti attraverso la rete neurale.

        Args:
            self (NeuralNetwork): Rete neurale.
            input_data (ndarray): Dati di input.

        Returns:
            ndarray: Uscita della rete neurale dopo la propagazione in avanti.
        """
        # Estrae i parametri della rete
        weights = self.layers_weights
        biases = self.layers_biases
        activation_functions = self.hidden_activation_functions
        num_layers = len(self.layers_weights)

        # Inizializza z con i dati di input
        z = input_data
        for layer in range(num_layers):
            # Calcola l'output del layer corrente
            a = np.matmul(weights[layer], z) + biases[layer]
            z = activation_functions[layer](a)

        return z

    def gradient_descent(self, learning_rate, weights_der, biases_der):
        """
        Aggiorna i pesi e i bias della rete neurale utilizzando il gradiente discendente.

        Args:
            self (NeuralNetwork): Rete neurale da aggiornare.
            learning_rate (float): Tasso di apprendimento per controllare la dimensione dei passi dell'aggiornamento.
            weights_der (list): Lista dei gradienti dei pesi per ciascuno strato.
            biases_der (list): Lista dei gradienti dei bias per ciascuno strato.

        Returns:
            NeuralNetwork: Rete neurale aggiornata.
        """
        for layer in range(len(self.layers_weights)):
            # Aggiornamento dei pesi utilizzando il gradiente discendente
            self.layers_weights[layer] -= learning_rate * weights_der[layer]

            # Aggiornamento dei bias utilizzando il gradiente discendente
            self.layers_biases[layer] -= learning_rate * biases_der[layer]

        return self

    def back_propagation(self, input_activations, layer_outputs, target, error_function):
        """
        Esegue la back-propagation per calcolare i gradienti dei pesi e dei bias.

        Args:
            self (NeuralNetwork): La rete neurale.
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

    def network_accuracy(self, input_data, labels):
        """
        Calcola l'accuratezza della rete neurale su un insieme di dati di input e target specificato.

        Args:
            self (NeuralNetwork): Oggetto della rete neurale.
            input_data (numpy.ndarray): Dati di input su cui valutare la rete.
            labels (numpy.ndarray): Target desiderati per i dati di input.

        Returns:
            float: Percentuale di predizioni corrette rispetto ai target desiderati.
        """
        output = self.forward_propagation(input_data)
        return ut.format_percentage(ut.compute_accuracy(output, labels))

    def train_neural_network(self, train_in, train_labels, validation_in, validation_labels, epochs=100,
                             learning_rate=0.1, rprop_type=RPropType.STANDARD):
        """
        Processo di apprendimento per la rete neurale.

        Args:
            self (NeuralNetwork): Rete neurale da addestrare.
            train_in (numpy.ndarray): Dati di input per il training.
            train_labels (numpy.ndarray): Target desiderati per i dati di input di training.
            validation_in (numpy.ndarray): Dati di input per la validazione.
            validation_labels (numpy.ndarray): Target desiderati per i dati di input di validazione.
            epochs (int, optional): Numero massimo di epoche per il training. Default: 100.
            learning_rate (float, optional): Tasso di apprendimento per il gradiente discendente. Default: 0.1.
            rprop_type (RPropType): Tipo di RProp da utilizzare (default: RpropType.STANDARD).

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
        weights_delta, biases_delta, weights_der_prev, biases_der_prev = None, None, None, None

        # Inizializzazione training
        train_net_out = self.forward_propagation(train_in)
        train_error = error_function(train_net_out, train_labels)
        train_errors.append(train_error)

        # Inizializzazione best_net
        validation_net_out = self.forward_propagation(validation_in)
        val_error = error_function(validation_net_out, validation_labels)
        validation_errors.append(val_error)

        min_val_error = val_error
        best_net = self.duplicate_network()

        train_accuracy = ut.compute_accuracy(train_net_out, train_labels)
        validation_accuracy = ut.compute_accuracy(validation_net_out, validation_labels)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        print(f'\n0/{epochs}\n'
              f'Training Accuracy: {ut.format_percentage(train_accuracy)}%,\n'
              f'Validation Accuracy: {ut.format_percentage(validation_accuracy)}%\n')

        # Inizio fase di apprendimento
        for epoch in range(epochs):
            # Gradient descent e Back-propagation
            layer_out, layer_act_fun_der = ut.compute_gradients(self, train_in)
            weights_der, biases_der = self.back_propagation(layer_act_fun_der, layer_out, train_labels, error_function)

            train_error_prev = 99999

            if epoch == 0:
                # Aggiornamento pesi tramite discesa del gradiente
                self.gradient_descent(learning_rate, weights_der, biases_der)

                # Inizializzazione dei pesi e dei bias per la funzione RProp
                weights_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in weights_der]
                biases_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in biases_der]

                weights_der_prev = deepcopy(weights_der)
                biases_der_prev = deepcopy(biases_der)
            else:
                # Aggiornamento della rete utilizzando la funzione RProp
                lf.rprops(self, weights_der, biases_der, weights_delta, biases_delta,
                               weights_der_prev, biases_der_prev, train_error, train_error_prev, rprop_type=rprop_type)

            if epoch > 0:
                train_error_prev = train_error

            # Forward propagation per training set
            train_net_out = self.forward_propagation(train_in)
            train_error = error_function(train_net_out, train_labels)
            train_errors.append(train_error)

            # Fase di validation
            validation_net_out = self.forward_propagation(validation_in)
            val_error = error_function(validation_net_out, validation_labels)
            validation_errors.append(val_error)

            # Trova l'errore minimo e la rete migliore
            if val_error < min_val_error:
                min_val_error = val_error
                best_net = self.duplicate_network()

            train_accuracy = ut.compute_accuracy(train_net_out, train_labels)
            validation_accuracy = ut.compute_accuracy(validation_net_out, validation_labels)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            print(f'\n{epoch + 1}/{epochs}\n'
                  f'Training Accuracy: {ut.format_percentage(train_accuracy)}%,\n'
                  f'Validation Accuracy: {ut.format_percentage(validation_accuracy)}%\n')

        ut.copy_params_in_network(self, best_net)

        return train_errors, validation_errors, train_accuracies, validation_accuracies
