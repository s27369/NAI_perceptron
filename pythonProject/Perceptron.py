from Util import *


class Perceptron:

    def __init__(self, num_inputs, learning_rate):
        self.weights = self.get_weights(num_inputs)
        self.bias = 1
        self.learning_rate = learning_rate

    def get_weights(self, amount):
        return [get_rand_nonzero() for _ in range(amount)]

    def activation(self, x):
        return 1 if x >= 0 else 0

    def get_delta(self, true_label, prediction):
        return true_label-prediction

    def dot_product(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"length of vectors does not match ({len(x)} vs {len(y)}): {x} and {y}")
        return sum([x[i] * y[i] for i in range(len(x))])

    def predict(self, observation):
        net = self.dot_product(observation[:-1], self.weights)
        return self.activation(net-self.bias)

    def little_train(self, observation):
        p = self.predict(observation)
        if p != observation[label_name]:
            delta = self.get_delta(observation[label_name], p)
            self.correct_bias(delta)
            self.correct_weights(observation, delta)
        return p



    def correct_weights(self,observation, delta):
        new_weights = [self.weights[i]+(delta*observation[i]*self.learning_rate) for i in range(len(self.weights))]
        self.weights = new_weights
    def correct_bias(self, delta):
        new_bias = self.bias+(delta*self.learning_rate)
        self.bias = new_bias
