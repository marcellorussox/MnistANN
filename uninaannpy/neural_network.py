from uninaannpy import activation_functions as af

import numpy as np


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

        self.layers_bias = []
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

        self.layers_bias.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons, 1)))

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

        biases = self.layers_bias
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
