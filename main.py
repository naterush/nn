import random
import math
from typing import List, Tuple
from collections import defaultdict

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

    @property
    def shape(self):
        return f'{self.rows}x{self.columns}'

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

def generate_training_data(size=1000) -> InputData:
    data = []
    for i in range(0, size):
        in_zero_to_200 = random.random() * 200
        greater_than_100 = in_zero_to_200 > 100
        data.append((Matrix([[in_zero_to_200], [in_zero_to_200]]), Matrix([[in_zero_to_200], [in_zero_to_200]])))
    random.shuffle(data)
    return data


def simoid_prime(num: float) -> float:
    return math.pow(math.e, -1 * num) / (math.pow(1 + math.pow(math.e, -1 * num), 2))

class Network:

    def __init__(self):
        self.num_layers = 3
        self.hidden_layer_height = 2
        
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

    def get_activatons(self, input: Matrix) -> List[Matrix]:
        """
        Returns a list where the element at an index is the activation at that
        layer in the network; includes input, hidden, and output layers
        """
        activations = [input]

        for i in range(self.num_layers):
            activations.append((self.weights[i] * activations[-1]).transform(sigmoid))

        return activations
        
    def cost(self, input_data: InputData) -> float:
        cost_arr = []
        for data in input_data:
            square_difference = (self.predict(data[0]) - data[1]).transform(lambda x: math.pow(x, 2))
            cost_arr.append(square_difference)

        return sum(cost_arr)

    def backprop(self, training_data: InputData) -> None:

        layer_to_weight_deltas = defaultdict(lambda: [])

        for data in training_data:
            activations = self.get_activatons(data[0])
            d_c_d_a = 2 * (activations[-1] - data[1])

            for layer in range(self.num_layers - 1, -1, -1):
                sigma_inverse = activations[layer].transform(lambda x: x * (1 - x))
                d_c_d_w = d_c_d_a * (activations[layer - 1] @ sigma_inverse).transpose()
                d_c_d_a = 2 * self.weights[layer - 1].transpose() * (d_c_d_a @ sigma_inverse)

                # TODO: should this be adjusting layer - 1 ? Or we are off by 1 above
                layer_to_weight_deltas[layer].append(d_c_d_w)

        for layer in range(self.num_layers):
            # TODO: introduce learning rate!
            weight_delta = sum(layer_to_weight_deltas[layer]) / len(training_data)
            self.weights[layer] -= weight_delta

                
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
    size = 10000
    training_data = generate_training_data(size)


    before_cost = network.cost(training_data) / size
    network.backprop(training_data)
    after_cost = network.cost(training_data) / size
    print(f'{before_cost=}, {after_cost=}')




if __name__ == '__main__':
    main()