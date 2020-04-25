import mlp_network as mlp
from mlp_tools import MultilayerDesign, TrainTestSplit
import numpy as np
import json
import os
from glob import glob


class InputReader(object):
    """InputReader.
    """

    def __init__(self, fname):
        """__init__.

        Parameters
        ----------
        fname :
            fname
        """
        with open(fname) as f:
            self.__dict__ = json.load(f)

    def get_keys(self):
        """get_keys.
        """
        return self.__dict__.keys()

    def get_values(self):
        """get_values.
        """
        return self.__dict__.values()

    def get_dict(self):
        """get_dict.
        """
        return self.__dict__


class RunMLP:
    """RunMLP.
    """

    def unit_run(
        crun,
        autosave,
        split,
        fnames,
        descrpt,
        act_func,
        epochs,
        l_rate,
        e_rate,
        min_error,
        network_layers,
    ):
        """unit_run.

        Parameters
        ----------
        crun :
            crun
        autosave :
            autosave
        split :
            split
        fnames :
            fnames
        descrpt :
            descrpt
        act_func :
            act_func
        epochs :
            epochs
        l_rate :
            l_rate
        e_rate :
            e_rate
        min_error :
            min_error
        network_layers :
            network_layers
        """
        for fname, descrp in zip(fnames, descrpt):
            for n_layer in network_layers:
                # Check for directory
                if autosave:
                    c_folder = RunMLP.check_or_make(
                        crun, n_layer[0], "{}-{}".format(split[0], split[1])
                    )
                    filenames = RunMLP.autonames(c_folder, descrp)
                # Load Data
                data = RunMLP.get_data(fname)
                # Check if test set is activated
                if not int(sum(split)):
                    trainingSet = data
                    testSet = data
                else:
                    trainingSet, testSet = TrainTestSplit.train_test_set(
                        data, ratio=split
                    )
                networkLayers = RunMLP.get_layerShape(data.shape[1], n_layer)

                RunMLP.mlp_run(
                    networkLayers=networkLayers,
                    activation=act_func,
                    learningRate=l_rate,
                    eta=e_rate,
                    minimumError=min_error,
                    maxNumEpochs=epochs,
                    trainingSet=trainingSet,
                    testSet=trainingSet,
                    log_filename_train=filenames[0],
                    log_filename_test=filenames[1],
                    tmp_filename="tmp.log",
                    save_name=filenames[2],
                )

                # print(fname, descrp, n_layer)

    @staticmethod
    def get_layerShape(witdh, n_layer):
        """get_layerShape.

        Parameters
        ----------
        witdh :
            witdh
        n_layer :
            n_layer
        """
        if n_layer[0] == "exp":
            return MultilayerDesign(witdh, n_layer[1]).exp_layer_stack()
        elif n_layer[0] == "square":
            return MultilayerDesign(witdh, n_layer[1]).square_layer_stack()
        elif n_layer[0] == "norm":
            return MultilayerDesign(witdh, n_layer[1]).norm_layer_stack()
        elif n_layer[0] == "cos":
            return MultilayerDesign(witdh, n_layer[1]).cosinus_layer_stack()
        elif n_layer[0] == "sin":
            return MultilayerDesign(witdh, n_layer[1]).sinus_layer_stack()
        else:
            print("Error in definition!")

    @staticmethod
    def check_or_make(*path):
        """check_or_make.

        Parameters
        ----------
        path :
            path
        """
        tmp_dir = os.path.join(*path)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    @staticmethod
    def autonames(path, names):
        """autonames.

        Parameters
        ----------
        path :
            path
        names :
            names
        """
        train_file = os.path.join(path, names + "_train.log")
        test_file = os.path.join(path, names + "_test.log")
        save_file = os.path.join(path, names + ".txt")
        return train_file, test_file, save_file

    @staticmethod
    def get_data(fname):
        """get_data.

        Parameters
        ----------
        fname :
            fname
        """
        return np.genfromtxt(fname, delimiter=",",)

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
        """mlp_run.

        Parameters
        ----------
        networkLayers :
            networkLayers
        activation :
            activation
        learningRate :
            learningRate
        eta :
            eta
        minimumError :
            minimumError
        maxNumEpochs :
            maxNumEpochs
        trainingSet :
            trainingSet
        testSet :
            testSet
        log_filename_train :
            log_filename_train
        log_filename_test :
            log_filename_test
        tmp_filename :
            tmp_filename
        save_name :
            save_name
        """

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


class SetupRun(InputReader):
    """SetupRun.
    """

    input_dict = {}
    run_key = []
    auto_save = False

    def __init__(self, fname):
        """__init__.

        Parameters
        ----------
        fname :
            fname
        """
        # super().__init__(fname)
        self.initialse(fname)

    def initialse(self, fname):
        """initialse.

        Parameters
        ----------
        fname :
            fname
        """
        ir = InputReader(fname)
        self.input_dict = ir.get_dict()
        self.run_key = ir.get_keys()

    def get_general(self, key):
        """get_general.

        Parameters
        ----------
        key :
            key
        """
        path = self.input_dict[key]["general_options"]["rpath"]
        autosave = self.input_dict[key]["general_options"]["autosave"]
        test = self.input_dict[key]["general_options"]["train_split"]
        return path, autosave, test

    def get_network(self, key):
        """get_network.

        Parameters
        ----------
        key :
            key
        """

        act_func = self.input_dict[key]["neural_network"]["function"]
        epochs = self.input_dict[key]["neural_network"]["epochs"]
        l_rate = self.input_dict[key]["neural_network"]["learning_rate"]
        e_rate = self.input_dict[key]["neural_network"]["error_rate"]
        min_error = self.input_dict[key]["neural_network"]["minimum_error"]
        network_layers = list(
            self.input_dict[key]["neural_network"]["network_layers"].items()
        )
        return act_func, epochs, l_rate, e_rate, min_error, network_layers

    @staticmethod
    def get_files(path):
        """get_files.

        Parameters
        ----------
        path :
            path
        """
        full_path = glob(path + "*csv")
        descrpt = [os.path.splitext(os.path.basename(fname))[0] for fname in full_path]
        return full_path, descrpt

    def setup_run(self):
        """setup_run.
        """

        for dkey in self.run_key:

            # Looding general setup for current run
            path, autosave, split = self.get_general(dkey)
            # Loading network setup for current run
            (
                act_func,
                epochs,
                l_rate,
                e_rate,
                min_error,
                network_layers,
            ) = self.get_network(dkey)

            fnames, descrpt = self.get_files(path)
            # Give it to the MPL class for running serial
            RunMLP.unit_run(
                crun=dkey,
                autosave=autosave,
                split=split,
                fnames=fnames,
                descrpt=descrpt,
                act_func=act_func,
                epochs=epochs,
                l_rate=l_rate,
                e_rate=e_rate,
                min_error=min_error,
                network_layers=network_layers,
            )


if __name__ == "__main__":
    # test_case_ii()
    # json_data = open('MLP/input_file.json', 'r')
    test1 = SetupRun("input_file.json")
    # print(test1.get_values())
    test1.setup_run()
