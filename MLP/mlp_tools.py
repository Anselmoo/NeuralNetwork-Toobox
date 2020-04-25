__all__ = ["MultilayerDesign", "TrainTestSplit"]
import numpy as np
from scipy.stats import norm


class MultilayerDesign:
    """MultilayerDesign.
    """

    witdh = 0
    n_layers = 0
    steps = 0

    def __init__(self, witdh, n_layers=0):
        """__init__.

        Parameters
        ----------
        witdh :
            witdh
        n_layers :
            n_layers
        """
        self.witdh = witdh - 1
        if n_layers:
            self.n_layers = n_layers
        else:
            self.n_layers = witdh
        self.steps = np.arange(self.n_layers)

    @staticmethod
    def l_round(x):
        """l_round.

        Parameters
        ----------
        x :
            x
        """

        layer_shape = np.round(x, 0).astype(int)
        layer_shape[np.argwhere(layer_shape == 0)] = 1

        return layer_shape

    @staticmethod
    def make_shape(func, n_layers, witdh):
        """make_shape.

        Parameters
        ----------
        func :
            func
        n_layers :
            n_layers
        witdh :
            witdh
        """

        layer_shape = np.zeros(n_layers)
        layer_shape[-1] = 1
        val = func
        off_set = np.abs(witdh - val[0])
        layer_shape[:-1] = val[:-1] + off_set
        return layer_shape

    def exp_layer_stack(self):
        """exp_layer_stack.
        """

        layer_shape = self.make_shape(np.exp(self.steps), self.n_layers, self.witdh)
        return self.l_round(layer_shape)

    def neg_exp_layer_stack(self):
        """neg_exp_layer_stack.
        """

        layer_shape = self.make_shape(
            self.witdh * np.exp(-self.steps), self.n_layers, self.witdh
        )
        return self.l_round(layer_shape)

    def square_layer_stack(self):
        """square_layer_stack.
        """

        layer_shape = self.make_shape(
            np.power(self.steps, 2), self.n_layers, self.witdh
        )
        return self.l_round(layer_shape)

    def neg_square_layer_stack(self):
        """neg_square_layer_stack.
        """

        layer_shape = self.make_shape(
            np.float_power(2, -self.steps), self.n_layers, self.witdh
        )
        return self.l_round(layer_shape)

    def log_layer_stack(self):
        """log_layer_stack.
        """

        layer_shape = self.make_shape(
            np.log(self.steps + self.witdh), self.n_layers, self.witdh
        )
        return self.l_round(layer_shape)

    def norm_layer_stack(self):
        """norm_layer_stack.
        """
        x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), self.n_layers)

        layer_shape = self.make_shape(
            np.multiply(self.witdh, norm.pdf(x)), self.n_layers, self.witdh
        )
        return self.l_round(layer_shape)

    def cosinus_layer_stack(self):
        """cosinus_layer_stack.
        """

        layer_shape = self.make_shape(
            np.abs(np.multiply(self.witdh, np.cos(self.steps))),
            self.n_layers,
            self.witdh,
        )
        return self.l_round(layer_shape)

    def sinus_layer_stack(self):
        """sinus_layer_stack.
        """

        layer_shape = self.make_shape(
            np.abs(np.multiply(self.witdh, np.sin(self.steps))),
            self.n_layers,
            self.witdh,
        )
        return self.l_round(layer_shape)


class TrainTestSplit:
    """TrainTestSplit.
    """

    @staticmethod
    def train_test_set(data, ratio):
        """train_test_set.

        Parameters
        ----------
        data :
            data
        ratio :
            ratio
        """

        cut_point = np.int(ratio[0] * data.shape[0] / ratio[1])
        return data[:cut_point], data[cut_point:]


class HyperOpt:
    """HyperOpt.
    """

    pass


if __name__ == "__main__":
    trainingSet = np.genfromtxt(
        "data/4_operators/6d5cb9b4b550b99515035260587fe41e5e94bd06ac35cadbf7e0de2f2ab8f92d.csv",
        delimiter=",",
    ).shape
    print(MultilayerDesign(trainingSet[1]).exp_layer_stack())
    print(MultilayerDesign(trainingSet[1]).neg_exp_layer_stack())
    print(MultilayerDesign(trainingSet[1]).square_layer_stack())
    print(MultilayerDesign(trainingSet[1]).neg_square_layer_stack())
    print(MultilayerDesign(trainingSet[1]).log_layer_stack())
    print(MultilayerDesign(trainingSet[1]).norm_layer_stack())
    print(MultilayerDesign(trainingSet[1]).cosinus_layer_stack())
    print(MultilayerDesign(trainingSet[1]).sinus_layer_stack())
    data = np.genfromtxt(
        "data/4_operators/6d5cb9b4b550b99515035260587fe41e5e94bd06ac35cadbf7e0de2f2ab8f92d.csv",
        delimiter=",",
    )
    print(TrainTestSplit.train_test_set(data, ratio=[2, 3]))
