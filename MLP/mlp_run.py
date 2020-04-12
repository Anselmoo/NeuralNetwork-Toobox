import mlp_network as mlp
import numpy as np
import json


class InputReader(object):
    def __init__(self, fname):
        with open(fname) as f:
            self.__dict__ = json.load(f)

    def get_keys(self):
        return self.__dict__.keys()

    def get_values(self):
        return self.__dict__.values()

    def get_dict(self):
        return self.__dict__


class SetupRun(InputReader):
    input_dict = {}
    run_key = []
    auto_save = False

    def __init__(self, fname):
        super().__init__(fname)
        self.initialse()

    def initialse(self):
        self.input_dict = self.get_dict()
        self.run_key = self.get_keys()

    def setup_run(self):

        for dkey in self.run_key:
            print(self.input_dict[dkey])

    def get_general():
        pass

    def get_network():
        pass

    @staticmethod
    def get_data():
        pass

    @staticmethod
    def mlp_run(
        networkLayers,
        activation,
        learningRate,
        eta,
        minimumError,
        maxNumEpochs,
        trainingSet,
        testSet,
        log_filename_train,
        log_filename_test,
        tmp_filename,
        save_name,
    ):

        backpropagation = mlp.Backpropagation(
            networkLayers=networkLayers,
            activation=activation,
            learningRate=learningRate,
            eta=eta,
            minimumError=minimumError,
            maxNumEpochs=maxNumEpochs,
        )

        backpropagation.train(
            trainingSet, log_filename=log_filename_train, tmp_filename=tmp_filename
        )
        backpropagation.test(
            testSet, log_filename=log_filename_test, tmp_filename=tmp_filename
        )
        backpropagation.save(save_name)

if __name__ == "__main__":
    # test_case_ii()
    # json_data = open('MLP/input_file.json', 'r')
    test1 = SetupRun("MLP/input_file.json")
    # print(test1.get_values())
    test1.setup_run()
