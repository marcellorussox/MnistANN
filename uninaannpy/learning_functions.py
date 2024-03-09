from enum import Enum

import numpy as np


class RPropType(Enum):
    STANDARD = 1
    RPROP_PLUS = 2
    IRPROP = 3
    IRPROP_PLUS = 4


def rprops(network, weights_der, biases_der, weights_delta, biases_delta, weights_der_prev,
           biases_der_prev, train_error, train_error_prev, eta_pos=1.2, eta_neg=0.5, delta_max=50, delta_min=0.00001,
           rprop_type=RPropType.STANDARD):
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
        train_error (float): Errore dell'epoca corrente.
        train_error_prev (float): Errore dell'epoca precedente.
        rprop_type (RPropType): Tipo di RProp da utilizzare (default: RpropType.STANDARD).

    Returns:
        NeuralNetwork: Rete neurale aggiornata con il metodo RProp.
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

                if weight_der_product > 0:
                    weights_delta[layer][num_rows][num_cols] = min(weights_delta[layer][num_rows][num_cols] * eta_pos,
                                                                   delta_max)

                    # Aggiornamento del delta del peso
                    layer_weights_delta[num_rows][num_cols] = -(np.sign(weights_der[layer][num_rows][num_cols]) *
                                                                weights_delta[layer][num_rows][num_cols])

                # Calcolo della nuova dimensione del delta per i pesi
                elif weight_der_product < 0:
                    weights_delta[layer][num_rows][num_cols] = max(weights_delta[layer][num_rows][num_cols] * eta_neg,
                                                                   delta_min)

                    if rprop_type != RPropType.STANDARD:
                        # Aggiornamento della derivata del peso
                        weights_der_prev[layer][num_rows][num_cols] = 0

                        if rprop_type != RPropType.IRPROP:
                            # Aggiornamento del delta del peso
                            layer_weights_delta[num_rows][num_cols] = 0

                            if (rprop_type == RPropType.IRPROP_PLUS and train_error > train_error_prev) or rprop_type == RPropType.RPROP_PLUS:
                                # Aggiornamento del delta del peso
                                layer_weights_delta[num_rows][num_cols] = -layer_weights_delta[num_rows][num_cols]
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
            if biases_der_product > 0:
                biases_delta[layer][num_rows][0] = min(biases_delta[layer][num_rows][0] * eta_pos, delta_max)

                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])

            elif biases_der_product < 0:
                biases_delta[layer][num_rows][0] = max(biases_delta[layer][num_rows][0] * eta_neg, delta_min)

                if rprop_type != RPropType.STANDARD:
                    # Aggiornamento della derivata del bias
                    biases_der_prev[layer][num_rows][0] = 0

                    if rprop_type != RPropType.IRPROP:
                        # Aggiornamento del delta del bias
                        layer_biases_delta[num_rows][0] = 0

                        if (rprop_type == RPropType.IRPROP_PLUS and train_error > train_error_prev) or rprop_type == RPropType.RPROP_PLUS:
                            # Aggiornamento del delta del bias
                            layer_biases_delta[num_rows][0] = -layer_biases_delta[num_rows][0]

            else:
                # Aggiornamento del delta del bias
                layer_biases_delta[num_rows][0] = -(np.sign(biases_der[layer][num_rows][0]) *
                                                    biases_delta[layer][num_rows][0])
            # Aggiornamento dei bias
            layer_biases[num_rows][0] += layer_biases_delta[num_rows][0]

            # Aggiornamento dei gradienti dei bias precedenti
            biases_der_prev[layer][num_rows][0] = biases_der[layer][num_rows][0]

    return network