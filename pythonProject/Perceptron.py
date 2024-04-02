from Util import *


class Perceptron:

    def __init__(self, num_inputs):
        self.weights = self.get_weights(num_inputs)
        self.bias = 1

    def get_weights(self, amount):
        return [get_rand_nonzero() for _ in range(amount)]

    def activation(self, x):
        return 1 if x >= 0 else 0

    def get_delta(self, true_label, prediction):
        return true_label-prediction;

    def dot_product(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"length of vectors does not match ({len(x)} vs {len(y)}): {x} and {y}")
        return sum([x[i] * y[i] for i in range(len(x))])

    def predict(self, observation):
        net = self.dot_product(observation, self.weights)
        return self.activation(net-self.bias)

    def little_train(self, observation):
        p = self.predict(observation)
        if p != observation[label_name]:
            pass
        return p

    def correct_weights(self, delta):
        pass
    def correct_bias(self, delta):
        pass
