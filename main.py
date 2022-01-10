import math
import random
from collections import defaultdict
from typing import List, Optional, Tuple


class Matrix:
    """
    A from-scratch matrix implementation, containing matrix multiplication, the hadamard product,
    transposing, and element-wise transformations.
    """

    def __init__(self, arr: Optional[List[List[float]]]=None, rows: int=0, columns: int=0, init_random: bool=False):
        """
        Parameters
        ----------
        arr
            A 2d array containing the starting values of the matrix. Must be rectangular.
        rows
            If no arr is passed, then the number of rows in the new matrix
        columns
            If no arr is passed, then the number of columns in the new matrix
        init_random
            If no arr is pased, then set to True if the new array should have random values between -.5 and .5. New values default to 0 otherwise.
        """
        if arr is None:
            arr = [[0 if not init_random else random.random() - .5] * columns for _ in range(rows)]

        # Check the array is 
        if not len({len(row) for row in arr}) <= 1:
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
        """Applies a function to all the elements of the matrix, indivigually."""
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                self.set_value(
                    row_index, column_index,
                    func(self.get_value(row_index, column_index))
                )
        return self

    def average(self):
        """Returns the average value in the matrix"""
        total_sum = 0
        for row_index in range(self.rows):
            row = self.get_row(row_index)
            total_sum += sum(row)
        return total_sum / (self.rows * self.columns)


    def __add__(self, other):
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

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(other * -1)

    def __mul__(self, other):
        """Standard matrix multiplication"""
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
        return self.__mul__(other)

    def __matmul__(self, other):
        """The hadamard product (e.g. element-wise muliplication). Uses the @ symbol."""

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

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        # make sure it's an int, and do piece wise division    
        int(other)

        result_matrix = Matrix(rows=self.rows, columns=self.columns)
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                result_matrix.set_value(row_index, column_index, self.get_value(row_index, column_index) / other)
        return result_matrix

    def __repr__(self) -> str:
        return str(self.arr)


InputData = List[Tuple[Matrix, Matrix]]


def sigmoid(x: float) -> float:
    """Implements 1 / (1 + e^(-x))"""
    try:
        return 1 / (1 + math.pow(math.e, -1 * x))
    except OverflowError:
        # If the number is too small, we get an overflow, and thus return 0
        return 0

def generate_training_data(size=1000) -> InputData:
    """Returns training data for the function > 100, over the range 0-200. Not super interesting, but it's ok."""
    data = []
    for i in range(0, size):
        in_zero_to_200 = random.random() * 200
        greater_than_100 = in_zero_to_200 > 100
        data.append((Matrix([[in_zero_to_200], [in_zero_to_200]]), Matrix([[in_zero_to_200], [in_zero_to_200]])))
    random.shuffle(data)
    return data


class Network:
    """A simple, rectangular neural network class with weights and no biases (coming soon (TM))."""

    def __init__(self):
        self.num_layers = 3
        self.hidden_layer_height = 2
        self.learning_rate = 1
        
        self.weights = [
            Matrix(rows=self.hidden_layer_height, columns=self.hidden_layer_height, init_random=True)
            for _ in range(self.num_layers)
        ]

    def predict(self, input: Matrix) -> Matrix:
        """Returns the activation of the final layer for a single input"""

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
        """The cost sum((y_hat - y)^2) over all input data."""
        cost_arr = []
        for data in input_data:
            square_difference = (self.predict(data[0]) - data[1]).transform(lambda x: math.pow(x, 2))
            cost_arr.append(square_difference.average())

        return sum(cost_arr) / len(input_data)

    def backprop(self, training_data: InputData) -> None:
        """
        Computes a weight delta for the training data, and updates the weights
        in the network. 

        Uses a version of backprop I derived with lots of help of the notes found
        here: https://cs231n.github.io/optimization-2/

        Lessons:
        1.  Start with the simple, non-vectorized example for the first step of the back-prop. It's pretty simple.
        2.  Then, do the same simple, non-vectorized example for the second to last layer of weights. 
            This is a bit harder, but not bad. 
        3.  Then, you can faily easily derive the vectorized examples! This is the best way to get to vectorized
            operations, methinks :-)

        It was really fun and super rewarding!
        """

        layer_to_weight_deltas = defaultdict(lambda: [])

        for data in training_data:
            activations = self.get_activatons(data[0])
            d_c_d_a = 2 * (activations[-1] - data[1])

            for layer in range(self.num_layers - 1, -1, -1):
                sigma_inverse = activations[layer].transform(lambda x: x * (1 - x))
                d_c_d_w = d_c_d_a * (activations[layer - 1] @ sigma_inverse).transpose()
                d_c_d_a = 2 * self.weights[layer - 1].transpose() * (d_c_d_a @ sigma_inverse)
                layer_to_weight_deltas[layer].append(d_c_d_w)

        for layer in range(self.num_layers):
            # TODO: introduce learning rate!
            weight_delta = sum(layer_to_weight_deltas[layer]) / len(training_data)
            self.weights[layer] -= self.learning_rate * weight_delta

                
def main():
    """Trains a simple, random network on the function > 100, and sees if the cost decreases (it does)!"""

    random.seed(10)
    network = Network()
    size = 10000
    training_data = generate_training_data(size)


    before_cost = network.cost(training_data)
    network.backprop(training_data)
    after_cost = network.cost(training_data)
    print(f'{before_cost=}, {after_cost=}')




if __name__ == '__main__':
    main()
