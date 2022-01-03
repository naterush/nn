import random
import math
from typing import List, Tuple


InputData = List[Tuple[float, float]]


def generate_training_data() -> InputData:
    data = []
    for i in range(-10000, 10000):
        data.append((i, 0 if i <= 100 else 1))
    random.shuffle(data)
    return data

def sigmoid(num: float) -> float:
    try:
        return 1 / (1 + math.pow(math.e, -1 * num))
    except OverflowError:
        # If the number is too small, we get an overflow, and thus return 0
        return 0

def simoid_prime(num: float) -> float:
    return math.pow(math.e, -1 * num) / (math.pow(1 + math.pow(math.e, -1 * num), 2))


class Network:

    def __init__(self):
        self.num_layers = 2
        self.weights = [random.random() - .5 for i in range(self.num_layers)]
        self.biases = [random.random() - .5 for i in range(self.num_layers)]

    def predict(self, input: float) -> float:
        current = sigmoid(input)
        for i in range(self.num_layers):
            current = sigmoid(current * self.weights[i] + self.biases[i])
        return current

    def feedforward(self, input: float) -> List[float]:
        """
        Returns a list of activations, with the length of the list equal to 
        self.num_layers + 1
        """
        activations = [sigmoid(input)]
        for i in range(self.num_layers):
            activations.append(sigmoid(activations[-1] * self.weights[i] + self.biases[i]))
        return activations


    def backprop(self, training_data: Tuple[float, float]) -> None:

        d_c_d_a_prev = 1
        activations = self.feedforward(training_data[0])

        for layer in range(self.num_layers - 1, -1, -1):
            print("HERE")
            # First, calculate the delta for the weight at that layer
            d_z_d_w = activations[layer]
            d_a_d_z = simoid_prime(activations[layer - 1] * self.weights[layer - 1] + self.biases[layer - 1])
            d_c_d_a = 2 * (activations[layer + 1] - training_data[0])
            weight_delta = d_z_d_w * d_a_d_z * d_c_d_a * d_c_d_a_prev
            
            # Then, calculate the bias for that delta
            d_z_d_b = 1
            bias_delta = d_z_d_b * d_a_d_z * d_c_d_a * d_c_d_a_prev

            # Finially, calculate the partial derivative of the cost function with respect the previous activation layer, so we 
            # can continue to back propagate
            d_z_d_a_prev =  self.weights[layer]
            d_c_d_a_prev = d_z_d_a_prev * d_a_d_z * d_c_d_a

            # TODO: append weight and bias to an array, so we can store them (and we can do this with d_c_d_a_prev as well?)
            # and then I am 
            print(f'{layer=}, {weight_delta=}, {bias_delta=}, {d_c_d_a_prev=}')


                

def main():
    """
    Here, we try and train an incredible vanilla, simple network 
    with 1 input layer, 1 hidden layer, and one output layer,
    and thus 3 total parameters, to learn the function > 100.

    To do 
    
    """

    training_data = generate_training_data()
    network = Network()

    network.backprop(training_data[0])

if __name__ == '__main__':
    main()