from Util import *
from Perceptron import Perceptron
import matplotlib.pyplot as plt
def discretize_dataset_labels(dataset):
    dataset[label_name] = [1 if x == "Iris-setosa" else 0 for x in dataset[label_name]]

def get_plot(accuracy):
    max_acc, min_acc = max(accuracy), min(accuracy)
    plt.plot( accuracy)
    plt.xlabel("Num of iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs num of iterations")
    plt.axhline(y=max_acc, color='green')
    plt.text(x=0, y=max_acc, s=f'Max accuracy: {max_acc}', color='green', fontsize=8, verticalalignment='bottom')
    plt.axhline(y=min_acc, color='red')
    plt.text(x=0, y=min_acc, s=f'Min accuracy: {min_acc}', color='red', fontsize=8, verticalalignment='bottom')
    plt.show()
def test_model(dataset, perceptron):
    predictions = []
    for i in range(len(dataset[label_name])):
        obs = get_observation(dataset, i)
        p = perceptron.predict(obs)
        predictions.append(p)
        print(f"Classified {'Setosa' if obs[-1]==1 else 'Non-setosa'} {'correctly' if p==obs[-1] else 'incorrectly'} ")
    acc = perceptron.get_accuracy(dataset, predictions)
    print(f"Accuracy={acc}")

def interface(test, perceptron):
    quit=False
    while not quit:
        print("Choose number:\n1 - input sample data to classify\n2 - train, test perceptron and get graph\n3 - quit\n>>>", end="")
        try:
            i = int(input())
        except:
            print("Incorrect input.")
            continue
        if i == 1:
            obs = input("input values separated by commas\n>>>")
            try:
                obs = obs.split(",")
                obs = [float(i.strip()) for i in obs]
                obs.append("unknown")
                print("Setosa" if perceptron.predict(obs)==1 else "Not-Setosa")
            except:
                print("incorrect input.")
        elif i == 2:
            try:
                if any(isinstance(x, str) for x in test[label_name]):
                    discretize_dataset_labels(test)
                test_model(test, perceptron)
            except:
                print("incorrect input")
        elif i == 3:
            return
        else:
            print("Incorrect input.")



if __name__ == '__main__':

    train, test = file_to_dict(r"data/iris_training.txt"), file_to_dict(r"data/iris_training.txt")
    perceptron = Perceptron(get_num_of_attributes(train), 0.2)

    discretize_dataset_labels(train)
    acc =perceptron.train(train, 100)
    # acc = perceptron.train(train)
    get_plot(acc)
    interface(train, test, perceptron)



