from typing import NamedTuple

import numpy as np

from utilities import activation_function_derivative, choose


class JordanNetwork:
    def __init__(self, alpha: float, error_max: float, iteration: int, num_amount_predict: int, matrix, activate: callable):
        self.a = alpha
        self.max_err = error_max
        self.iteration_threshold = iteration
        self.num_amount_predict = num_amount_predict
        self.matrix = matrix
        self.weights1l = np.random.uniform(
            -1,
            1,
            (self.matrix.shape[1], self.matrix.shape[0])
        )
        self.weights2l = np.random.uniform(
            -1,
            1,
            (self.matrix.shape[0], 1)
        )
        self.context_neurons = np.zeros((1, 1))
        self.error = 0
        self.run = NamedTuple(
            'Run',
            inputs=np.ndarray,
            target=np.ndarray,
            first_net=np.ndarray,
            first_out=np.ndarray,
            second_net=np.ndarray,
            second_out=np.ndarray
        )
        self.activation_fun = activate

    def train_network(self):
        error = 1e25
        while error > self.max_err and self.iteration_threshold > 0:
            error = 0
            self.iteration_threshold = self.iteration_threshold - 1
            self.teardown_context_neurons()
            np.apply_along_axis(self.vector_train, 1, self.matrix)
            self.teardown_context_neurons()
            error_vec = np.apply_along_axis(self.calc_error, 1, self.matrix)
            for i, element in np.ndenumerate(error_vec):
                error = error + element
            print("Error = ", error)
            self.error = error
        print(self.iteration_threshold, 'iters')
        self.predict()

    def predict_next_seq_val(self):
        self.teardown_context_neurons()
        np.apply_along_axis(self.process_vector, 1, self.matrix)
        last_num = self.matrix[self.matrix.shape[0] - 1]
        last_num[last_num.size - 1] = 0
        i = 0
        print("Matrix \n", self.matrix)
        print("First layer weights \n", self.weights1l)
        print("Second layer weights\n", self.weights2l)
        while i < self.num_amount_predict:
            second_out = self.process_vector(last_num)
            print("Next num:", second_out)
            last_num[last_num.size - 1] = second_out
            last_num = last_num[1:]
            last_num = np.append(last_num, 0)
            i = i + 1

    def teardown_context_neurons(self):
        self.context_neurons = np.zeros((1, 1))

    def run_input_vector_through_net(self, start_vector):
        input_vector = start_vector[:start_vector.size - 1]
        target_value = start_vector[start_vector.size - 1]
        input_vector = np.append(input_vector, self.context_neurons)
        first_net = np.matmul(input_vector, self.weights1l)
        first_out = self.activation_fun(first_net)
        second_net = np.matmul(first_out, self.weights2l)
        second_out = self.activation_fun(second_net)
        self.context_neurons = second_out
        return self.run(
            inputs=input_vector,
            target=target_value,
            first_net=first_net,
            first_out=first_out,
            second_net=second_net,
            second_out=second_out
        )

    def calc_error(self, vector):
        run = self.run_input_vector_through_net(vector)
        return 1 / 2 * ((run.target - run.second_out[0]) ** 2)

    def vector_train(self, vector):
        run = self.run_input_vector_through_net(vector)
        der_weights_first = self.get_derivative_weights_first(
            run.target,
            run.second_out[0],
            run.second_net[0],
            run.first_net,
            run.inputs
        )
        self.weights1l = self.weights1l - self.a * der_weights_first
        der_weights_second = self.get_derivative_weights_second(
            run.target,
            run.second_out[0],
            run.second_net[0],
            run.first_out
        )
        self.weights2l = self.weights2l - self.a * der_weights_second

    def process_vector(self, vector):
        run = self.run_input_vector_through_net(vector)
        return run.second_out[0]

    def get_derivative_weights_second(self, target, second_out, second_net, first_out):
        derivative_weights_second = np.zeros(
            (
                self.weights2l.shape[0],
                self.weights2l.shape[1]
            )
        )
        for (i, j), element in np.ndenumerate(derivative_weights_second):
            derivative_weights_second[i, j] = -1 * (target - second_out) \
                                              * activation_function_derivative(second_net) \
                                              * first_out[j]
        return derivative_weights_second

    def get_derivative_weights_first(self, target, second_out, second_net, first_net, inputs):
        derivative_weights_first = np.zeros(
            (
                self.weights1l.shape[0],
                self.weights1l.shape[1]
            )
        )
        for (i, j), element in np.ndenumerate(derivative_weights_first):
            derivative_weights_first[i, j] = -1 * (target - second_out) \
                                             * activation_function_derivative(second_net) \
                                             * self.weights2l[j, 0] \
                                             * activation_function_derivative(first_net[j]) * inputs[i]
        return derivative_weights_first

    def save_weight_matrices(self, seq_name):
        np.save(f"weights/{choose(seq_name)}1.npy", self.weights1l)
        np.save(f"weights/{choose(seq_name)}2.npy", self.weights2l)

    def load_weight_matrices(self, seq_name):
        self.weights1l = np.load(f"weights/{choose(seq_name)}1.npy")
        self.weights2l = np.load(f"weights/{choose(seq_name)}2.npy")
