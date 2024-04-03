from Util import *
from Perceptron import Perceptron

def discretize_dataset_labels(dataset):
    dataset[label_name] = [1 if x == "Iris-setosa" else 0 for x in dataset[label_name]]

def test_model(dataset, perceptron):
    predictions = []
    for i in range(len(dataset[label_name])):
        obs = get_observation(dataset, i)
        p = perceptron.predict(obs)
        predictions.append(p)
        print(f"Classified {'Setosa' if obs[-1]==1 else 'Non-setosa'} {'correctly' if p==obs[-1] else 'incorrectly'} ")
    acc = perceptron.get_accuracy(dataset, predictions)
    print(f"Accuracy={acc}")

if __name__ == '__main__':
    train, test = file_to_dict(r"data/iris_training.txt"), file_to_dict(r"data/iris_training.txt")
    perceptron = Perceptron(get_num_of_attributes(train), 0.2)

    discretize_dataset_labels(train)
    # perceptron.train(train, 500)
    perceptron.train(train)

    discretize_dataset_labels(test)
    test_model(test, perceptron)

