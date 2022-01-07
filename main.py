import random
import math
from typing import List, Tuple
from collections import defaultdict
from copy import deepcopy

def is_rectangular(arr):
    return len({len(row) for row in arr}) <= 1

class Matrix:

    def __init__(self, arr=None, rows=0, columns=0, init_random=False):
        if arr is None:
            arr = [[0 if not random else random.random() - .5] * columns for _ in range(rows)]

        if not is_rectangular(arr):
            raise ValueError(f'Arr is not rectangular: {arr}')

        self.arr = arr
        self.rows = len(arr)
        self.columns = len(arr[0]) if self.rows > 0 else 0

    def get_value(self, row_index, column_index):
        return self.arr[row_index][column_index]

    def set_value(self, row_index, column_index, value):
        self.arr[row_index][column_index] = value

    def get_column(self, column_index):
        column = []
        for row in self.arr:
            column.append(row[column_index])
        return column

    def get_row(self, row_index):
        row_index_int = int(row_index)
        return self.arr[row_index_int]

    def transpose(self):
        # Turn the columns in the current matrix into rows in the new matrix
        new_arr = [self.get_column(column_index) for column_index in range(self.columns)]
        return Matrix(new_arr)

    def transform(self, func):
        """
        Applies a function to all the elements of the matrix
        """
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                self.set_value(
                    row_index, column_index,
                    func(self.get_value(row_index, column_index))
                )
        return self

    def __mul__(self, other):

        if isinstance(other, Matrix):
            if self.columns != other.rows:
                raise ValueError(f'Cannot mulitply matrix with dimensions {self.rows=} x {self.columns=} and {other.rows=}, {other.columns=}')

            result_matrix = Matrix(rows=self.rows, columns=other.columns)
            for row_index in range(self.rows):
                for column_index in range(len(other.arr[0])):
                    column = other.get_column(column_index)
                    row = self.get_row(row_index)
                    # Pairwise mulitple them (dot product)
                    result = sum(a * b for a, b in zip(row, column))
                    result_matrix.set_value(row_index, column_index, result)

            return result_matrix
        else:
            int(other)

            result_matrix = Matrix(rows=self.rows, columns=self.columns)
            for row_index in range(self.rows):
                for column_index in range(self.columns):
                    new_value = self.get_value(row_index, column_index) * other
                    result_matrix.set_value(row_index, column_index, new_value)

            return result_matrix

    def __rmul__(self, other):
        # TODO: check this works correctly (e.g. mul should work first)
        return self.__mul__(other)

    
    def __add__(self, other):

        # 2 + m
        # m + m
        result_matrix = Matrix(rows=self.rows, columns=self.columns)

        if isinstance(other, Matrix):
            # TODO: checking the same size
            for row_index in range(self.rows):
                for column_index in range(self.columns):
                    new_value = self.get_value(row_index, column_index) + other.get_value(row_index, column_index)
                    result_matrix.set_value(row_index, column_index, new_value)

            return result_matrix
        else:
            # Check it's a number (this will throw an error if it's a number)
            int(other)

            # TODO: check this is a number
            for row_index in range(self.rows):
                for column_index in range(self.columns):
                    new_value = self.get_value(row_index, column_index) + other
                    result_matrix.set_value(row_index, column_index, new_value)
    
            return result_matrix

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError(f'Can only matrix mulitply two different matrixes, not {other}')

        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError(f'Matrix mulitplication requires the same dimensions, not {self.rows=} x {self.columns=} and {other.rows=} x {other.columns=}')

        result_matrix = Matrix(rows=self.rows, columns=self.columns)
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                result_matrix.set_value(
                    row_index, column_index,
                    self.get_value(row_index, column_index) * other.get_value(row_index, column_index)
                )
        return result_matrix

    def __truediv__(self, other):
        # make sure it's an int, and do piece wise division    
        int(other)

        result_matrix = Matrix(rows=self.rows, columns=self.columns)
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                result_matrix.set_value(row_index, column_index, self.get_value(row_index, column_index) / other)
        return result_matrix


    def __rmatmul__(self, other):
        return self.__matmul__(other)


    def __sub__(self, other):
        return self.__add__(other * -1)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self) -> str:
        return str(self.arr)


InputData = List[Tuple[Matrix, Matrix]]

def sigmoid(num: float) -> float:
    try:
        return 1 / (1 + math.pow(math.e, -1 * num))
    except OverflowError:
        # If the number is too small, we get an overflow, and thus return 0
        return 0

def generate_training_data() -> InputData:
    data = []
    for i in range(0, 200):
        data.append((Matrix([[i], [i], [i]]), Matrix([[sigmoid(i)], [sigmoid(i)], [sigmoid(i)]])))
    random.shuffle(data)
    return data


def simoid_prime(num: float) -> float:
    return math.pow(math.e, -1 * num) / (math.pow(1 + math.pow(math.e, -1 * num), 2))

class Network:

    def __init__(self):
        # We have 2 hidden layers with 3 nodes in them each; so including input and output 
        # we have a total of 4 layers
        self.num_layers = 4
        self.hidden_layer_height = 3
        
        self.weights = [
            Matrix(rows=self.hidden_layer_height, columns=self.hidden_layer_height, init_random=True)
            for _ in range(self.num_layers)
        ]

    def predict(self, input: Matrix) -> float:
        current = input

        for i in range(self.num_layers):
            # Calculate all the hidden edges, and apply the sigmoid at the very end
            current = (self.weights[i] * current).transform(sigmoid)

        return current

    def feedforward(self, input: Matrix) -> Tuple[List[Matrix], List[Matrix]]:
        """
        Returns a list of the input to the sigmoid function (we call this Z or D). Also
        a list of the activations at each layer.
        """
        activations = [input]
        before_sigma = []

        for i in range(self.num_layers):
            # Calculate all the hidden edges, and apply the sigmoid at the very end
            before_sigma.append(self.weights[i] * activations[-1])
            activations.append(before_sigma[-1].transform(sigmoid))

        return activations, before_sigma
        
    def cost(self, input_data: InputData) -> float:
        cost_arr = []
        for data in input_data:
            cost_arr.append(math.pow(self.predict(data[0]) - data[1], 2))

        return sum(cost_arr)

    def backprop(self, training_data: InputData) -> None:

        layer_to_weight_deltas = defaultdict(lambda: [])

        for data in training_data:
            grad = Matrix([[1], [1], [1]])
            activations, before_sigma = self.feedforward(data[0])

            for layer in range(self.num_layers - 1, -1, -1):
                dAdD = activations[layer].transform(lambda x: x * (1 - x)) @ grad
                dAdX = (self.weights[layer - 1].transpose() * before_sigma[layer - 1]) @ dAdD
                dAdW = (before_sigma[layer - 1] * activations[layer - 1].transpose()) * dAdD

                # TODO: should this be adjusting layer - 1 ? Or we are off by 1 above
                layer_to_weight_deltas[layer].append(dAdW)
                grad = dAdX

        for layer in range(self.num_layers):
            # TODO: introduce learning rate!
            weight_delta = sum(layer_to_weight_deltas[layer]) / len(training_data)
            bias_delta = sum(layer_to_weight_deltas[layer]) / len(training_data)

            print(weight_delta)
            self.weights[layer] += weight_delta
            self.biases[layer] += bias_delta

            print(f'Updated {layer} by {weight_delta=}, {bias_delta=}')

                
def main():
    """
    Here, we try and train an incredible vanilla, simple network 
    with 1 input layer, 1 hidden layer, and one output layer,
    and thus 3 total parameters, to learn the function > 100.

    Cluster points in 2d; decide above and below in a line.

    TODO:
    -   1 - 3 - 1
    -   I should be able to just shove in 0 - 200; if you have one input, you don't
        need to worry; If you have multiple features, you have to make sure they are
        standized in some way... dont' worry about it for a single feature.
    -   Train for a certain cost (rather than a certain size) - some threshold when you
        stop training (when the weights aren't moving much). 
    """

    random.seed(10)
    network = Network()

    input = Matrix(arr=[[40], [40], [40]])

    #print(f'{network.predict(input)=}')
    #print(f'{network.feedforward(input)=}')

    training_data = generate_training_data()

    network.backprop(training_data)




if __name__ == '__main__':
    main()