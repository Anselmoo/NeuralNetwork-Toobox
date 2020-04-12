import mlp_network as mlp
import numpy as np


def test_case_i():
    networkLayer = [3, 5, 1]

    # feedForward = FeedForward(networkLayer, "Sigmoid")

    backpropagation = mlp.Backpropagation(
        networkLayer, "Sigmoid", 0.7, 0.5, maxNumEpochs=100000
    )

    trainingSet = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
        ]
    )

    backpropagation.initialise()
    result = backpropagation.train(trainingSet)
    backpropagation.test(trainingSet)


def test_case_ii():
    networkLayer = [5, 9, 3, 1]

    # feedForward = FeedForward(networkLayer, "Sigmoid")

    backpropagation = mlp.Backpropagation(
        networkLayer, "Sigmoid", 0.7, 0.5, maxNumEpochs=100000
    )
    trainingSet = np.genfromtxt(
        "data/4_operators/6d5cb9b4b550b99515035260587fe41e5e94bd06ac35cadbf7e0de2f2ab8f92d.csv",
        delimiter=",",
    )
    print(trainingSet)
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)
    backpropagation.test(trainingSet)


if __name__ == "__main__":
    test_case_ii()
