import numpy as np
import activationFunctions as af


class NeuralNetwork:
    MU, SIGMA = 0, 0.1

    # Costruttore per la creazione della rete neurale
    def __init__(self, hidden_activation_functions, output_activation_function, error_function,
                 input_layer_size, hidden_layers, output_layer_size):

        self.layers_bias = []
        self.layers_weights = []
        self.hidden_layers = hidden_layers

        if len(hidden_activation_functions) != len(hidden_layers):
            raise ValueError("The number of hidden activation function must be equal to the number of hidden layers")

        self.hidden_activation_functions = hidden_activation_functions
        hidden_activation_functions.append(output_activation_function)
        self.error_function = error_function
        self.number_of_hidden_layers = len(hidden_layers)
        self.__initialize_parameters(input_layer_size, output_layer_size)

    # Inizializzazione dei parametri
    def __initialize_parameters(self, input_layer_size, output_layer_size):
        hidden_layer_size = self.hidden_layers
        number_of_hidden_layers = self.number_of_hidden_layers

        self.__initialize_weights(0, hidden_layer_size[0], input_layer_size)
        self.__initialize_bias(0, hidden_layer_size[0])

        for i in range(1, self.number_of_hidden_layers):
            self.__initialize_weights(i, hidden_layer_size[i], hidden_layer_size[i])
            self.__initialize_bias(i, hidden_layer_size[i])

        self.__initialize_weights(number_of_hidden_layers, output_layer_size,
                                  hidden_layer_size[number_of_hidden_layers - 1])
        self.__initialize_bias(number_of_hidden_layers, output_layer_size)

    # Inizializzazione dei pesi
    def __initialize_weights(self, index, number_of_layer_neurons, input_variables):
        # I numeri sono generati dalla distribuzione standard (gaussiana)
        self.layers_weights.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons,
                                                                                      input_variables)))

    # Inizializzazione dei bias
    def __initialize_bias(self, index, number_of_layer_neurons):
        # I numeri sono generati dalla distribuzione standard (gaussiana)
        self.layers_bias.insert(index, np.random.normal(self.MU, self.SIGMA, size=(number_of_layer_neurons, 1)))

    # Funzione per ottenere i pesi della rete
    def get_weights(network, i=0):
        W = network.layers_weights
        if (i > 0):
            return W[i - 1]
        else:
            return W

    # Funzione per ottenere i bias della rete
    def get_biases(network, i=0):
        B = network.layers_bias
        if (i > 0):
            return B[i - 1]
        else:
            return B

    # Funzione per ottenere le funzioni di attivazione
    def get_act_fun(network, i=0):
        AF = network.hidden_activation_functions
        if (i > 0):
            return AF[i - 1]
        else:
            return AF

    # Funzione per modificare le funzioni di attivazione
    def set_activation_function(network, layer_indices=[], activation_function=af.tanh, layer_type=1):
        if layer_type == 0:  # Modifica la funzione di attivazione per gli strati specificati
            if np.isscalar(layer_indices):
                network.hidden_activation_functions[layer_indices - 1] = activation_function
            else:
                count = 0
                for i in layer_indices:
                    network.hidden_activation_functions[i - 1] = activation_function[count]
                    count += 1
        elif layer_type == 1:  # Modifica al funzione di attivazione per tutti gli strati interni
            for i in range(network.number_of_hidden_layers):
                network.hidden_activation_functions[i] = activation_function
        else:  # Modifica la funzione di attivazione per lo strato di output
            network.hidden_activation_functions[network.number_of_hidden_layers] = activation_function

        # Controllo per verificare se il numero di funzioni di attivazione è uguale al numero di strati interni
        if len(network.hidden_activation_functions) != network.number_of_hidden_layers + 1:
            raise ValueError("The number of hidden activation function must be equal to the number of hidden layers")

        return network
