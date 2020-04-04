import numpy as np
import sys
import logging
import os


class HyperbolicTangent:
    def getActivation(self, net):
        return np.tanh(net)

    def getDerivative(self, net):
        return 1 - (self.getActivation(net) * self.getActivation(net))


class Sigmoid:
    def getActivation(self, net):
        return 1 / (1 + np.exp(-1 * net))

    def getDerivative(self, net):
        return self.getActivation(net) * (1 - self.getActivation(net))


class FeedForward(HyperbolicTangent, Sigmoid):
    """FeedForward"""

    networkLayers = np.array([])
    activation = ""
    totalNumNodes = 0
    net = np.array([])
    weights = np.array([])
    biasWeights = np.array([])
    values = np.array([])

    def __init__(self, networkLayers, activation, dtype=np.float64):
        """__init__.
       
        Parameters
        ----------
        networkLayers : list
            pattern of the number of input-, hidden-, and output-layers
        activation : str
            type of the activation function
        dtype : data type objects, optional
            The data-type objects (dtype) can be set to single (np.float32) or double (np.float64) precission, by default np.float64       
        """
        self.networkLayers = []
        startNode = 0
        endNode = 0
        for layer, numNodes in enumerate(networkLayers):
            if layer > 0:
                startNode += networkLayers[layer - 1]
            endNode += numNodes
            self.networkLayers.append(
                {
                    "num_nodes": numNodes,
                    "start_node": startNode,
                    "end_node": endNode - 1,
                }
            )

        self.totalNumNodes = np.sum(networkLayers, dtype=np.int)

        if activation == "Sigmoid":
            self.activation = Sigmoid()
        elif activation == "HyperbolicTangent":
            self.activation = HyperbolicTangent()
        else:
            print("Model not implemented")
            sys.exit(0)

        self.dtype = dtype

    def initialise(self):
        """initialise bias-neurons and their weights.
        
        Bias-neurons- and their weights-matrices will be initialise by using np.zeros. The data-type (dtype) can be set 
        in the __init__.py, and can be choosen between single (np.float32) and double (np.float64) precission.
        """
        self.weights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )

        self.initialiseValuesNet()
        self.initialiseWeights()

    def initialiseValuesNet(self):
        """initialiseValuesNet values- and  net-array.
        """

        self.values = np.zeros(self.totalNumNodes, dtype=self.dtype)
        self.net = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def initialiseWeights(self, low=-5, high=+5):
        """initialiseWeights for the weights and the bias-weights.
        
        Based on the zero-matrix, the weights- and bias-weights-matrix will be filled with random-int, which
        becomes float by np.divide. 
        
        Notes
        -----
        np.divide is important because it will also keep the dtype-format consistent.
        
        Parameters
        ----------
        low : int, optional
            lowest random-value, by default -5
        high : int, optional
            highes random-value, by default +5
        """

        self.weights = np.divide(
            np.random.randint(low=low, high=high, size=self.weights.shape),
            100.0,
            dtype=self.dtype,
        )
        self.biasWeights = np.divide(
            np.random.randint(low=low, high=high, size=self.biasWeights.shape),
            100.0,
            dtype=self.dtype,
        )

    def activate(self, inputs):
        """activate the forward propagation.
        
        For activate the forward propagation, the values(dendrites) will be elementwise combined with the 
        weights (synapes) plus the bias. This will be performed by numpy. Furthermore, the activation functions
        will be activated to the values.
        
        Notes:
        ------
        A more detail description is provided by Deep Learning: Ian Goodfellow et al. page 205
        
        Parameters
        ----------
        inputs : array
            inputs as float array to be processed
        """
        # Defining the h^0 = x
        _end = self.networkLayers[0]["num_nodes"]
        self.values[0:_end] = inputs[0:_end]

        # Connecting the layers (input, hidden, and target) via j-index
        for k, layer in enumerate(self.networkLayers[1:]):
            # Prevouis layer
            i_0, i_1 = self.getPreviousLayer(self.networkLayers, index=k)

            # Current layer
            j_0, j_1 = self.getCurrentLayer(layer)

            # Apply feedback transformation

            self.net[j_0:j_1] = np.add(
                self.biasWeights[k, j_0:j_1],
                np.dot(self.values[i_0:i_1], self.weights[i_0:i_1, j_0:j_1]),
            )

            # Apply activation function
            self.values[j_0:j_1] = self.activation.getActivation(self.net[j_0:j_1])

    def getOutputs(self):
        """getOutputs returns the predicted values
        
        Returns
        -------
        array
            Returns the predicted values as float array
        """
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        return self.values[startNode : endNode + 1]

    def getNetworkLayers(self):
        """getNetworkLayers.
        
        Returns
        -------
         : dict 
            Dictonary of the network-layers including total-, start-, and end-number of nodes.
        """
        return self.networkLayers

    def getValue(self):
        """getValue.
        
        Returns
        -------
         : array
            All values as float-array
        """
        return self.values

    def getValueEntry(self, index):
        """getValueEntry.
        
        Returns the values for a explitic list of array-indices
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
        : array
            List of crop array-entries as array
        """
        return self.values[index]

    def getActivation(self):
        """getActivation [summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.activation

    def getNet(self):
        """getNet returns the net.
        
        Returns
        -------
        : array
            float-array of the net
        """
        return self.net

    def getNetEntry(self, index):
        """getNetEntry entry-value of the net.
        
        getNetEntry returns the current entry of the net based on the current index.
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
         : float 
            entry of the net based on the index
        """
        return self.net[index]

    def getWeight(self):
        """getWeight return the weight.
        
        Returns
        -------
        : array
            float-array of the weights
        """
        return self.weights

    def getWeightEntry(self, index):
        """getWeightEntry entry-value of the weight.
        
        [extended_summary]
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
         : float 
            entry of the net based on the index
        """
        return self.weights[index]

    def getBiasWeight(self):
        """
        getBiasWeights [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights

    def getBiasWeightEntire(self, index):
        """
        getBiasWeightEntire [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights[index]

    def setBiasWeights(self, biasWeights):
        """
        setBiasWeights [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        biasWeights : [type]
            [description]
        """
        self.biasWeights = biasWeights

    def updateWeight(self, i, j, weight):
        """updateWeight
        
        Parameters
        ----------
        i : list
            int-list of the column-entries
        j : list
            int-list of the row-entries
        weight : array
            updated weight-coefficients
        """
        # self.weights[i, j] += weight
        self.weights[i[0] : i[1], j[0] : j[1]] = np.add(
            self.weights[i[0] : i[1], j[0] : j[1]], weight, dtype=self.dtype
        )

    def updateBiasWeight(self, i, j, weight):
        """
        updateBiasWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        i : list
            int-list of the column-entries
        j : list
            int-list of the row-entries
        weight : array
             updated bias-weight-coefficients
        """
        # self.biasWeights[i, j] += weight
        if isinstance(i, int):
            self.biasWeights[i, j[0] : j[1]] = np.add(
                self.biasWeights[i, j[0] : j[1]], weight, dtype=self.dtype
            )
        else:
            self.biasWeights[i[0] : i[1], j[0] : j[1]] = np.add(
                self.biasWeights[i[0] : i[1], j[0] : j[1]], weight, dtype=self.dtype
            )

    def getTotalNumNodes(self):
        """getTotalNumNodes.
        
        Returns
        -------
         : int
            Total number of nodes as int
        """
        return self.totalNumNodes

    def getDtype(self):
        """getDtype."""
        return self.dtype

    def save(self, filename):
        """save the trained MLP-network.
        
        Saved the FeedForward-class (MLP-network) as binary `pickle`-file.
        
        Parameters
        ----------
        filename : str
            filename of the to save pickle-file
        """
        with open(filename, "wb") as network_file:
            pickle.dump(self, network_file)

    @staticmethod
    def getPreviousLayer(layer, index, ext=+1):
        """getPreviousLayer start and end point of previous layer.
        
        Parameters
        ----------
        layer : dict
            Dictonary of the previous layer
        index : int
            index to select of the previous layer
        ext: int, optional
            Extend the layer-range, by default +1

        
        Returns
        -------
        i_0 : int
            startpoint of the previous layer
        
        i_1 : int
            endpoint of the previous layer
        """
        i_0, i_1 = layer[index]["start_node"], layer[index]["end_node"] + ext
        return i_0, i_1

    @staticmethod
    def getCurrentLayer(layer):
        """getCurrentLayer start and end point of current layer.
        
        Parameters
        ----------
        layer : dict
            Dictonary of the current layer
        
        Returns
        -------
        i_0 : int
            startpoint of the current layer
        
        i_1 : int
            endpoint of the current layer
        """
        i_0, i_1 = layer["start_node"], layer["end_node"] + 1
        return i_0, i_1

    @staticmethod
    def load(filename):
        """load a trained MLP-network.
        
        load a pre-trained FeedForward-class (MLP-network) from binary `pickle`-file.
        
        Parameters
        ----------
        filename : str
            filename of the to load pickle-file
        
        Returns
        -------
         : array
            gives back the weight and bias coefficients of the MLP-network
        """
        with open(filename, "rb") as network_file:
            network = pickle.load(network_file)
            return network


class Backpropagation(FeedForward):
    """Backpropagation
    
    The Backpropagation class calculates the minimum value of the error function in relation to the training-set and the activation function.
    The technique for achieving this goal is called the delta rule or gradient descent. 
    
    """

    nodeDeltas = np.array([])
    gradients = np.array([])
    biasGradients = np.array([])
    learningRate = np.array([])
    eta = np.array([])
    weightUpdates = np.array([])
    biasWeightUpdates = np.array([])
    minimumError = ""
    maxNumEpochs = ""
    numEpochs = ""
    network = np.array([])
    delta = np.float64
    networkLayers = []
    error = 0.0

    def __init__(
        self,
        networkLayers,
        activation,
        learningRate,
        eta,
        minimumError=0.005,
        maxNumEpochs=2000,
        dtype=np.float64,
    ):
        """__init__.
        
        Parameters
        ----------
        networkLayers : list
            pattern of the number of input-, hidden-, and output-layers
        activation : str
            type of the activation function   
        learningRate : float
            Learning rate of the MLP
        eta : float
            Error correction factor
        minimumError : float, optional
            Minimal error to stop the training, by default 0.005
        maxNumEpochs : int, optional
            Maxinum numbers of epochs before stopping the training, by default 2000
        dtype : data type objects, optional
            The data-type objects (dtype) can be set to single (np.float32) or double (np.float64) precission, by default np.float64
        """
        self.network = FeedForward(
            networkLayers=networkLayers, activation=activation, dtype=dtype
        )
        self.learningRate = learningRate
        self.eta = eta
        self.minimumError = minimumError
        self.maxNumEpochs = maxNumEpochs
        self.initialise()

    def initialise(self):
        """initialise MLP.
        
        The intiale procedure includes:
            1. network
            2. node deltas
            3. gradients of values
            4. gradients of bias
            5. Update matrices for:
                a. weight of values
                b. weight of bias
                c. gradients of values
                d. gradients of bias
             
        """
        self.network.initialise()
        self.nodeDeltas = np.array([])
        self.gradients = np.array([])
        self.biasGradients = np.array([])
        self.totalNumNodes = self.network.getTotalNumNodes()
        self.dtype = self.network.getDtype()
        self.networkLayers = self.network.getNetworkLayers()
        # initiale the weight, bias, and gradients matrices
        self.weightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.gradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasGradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.initialiseValues()

    def initialiseValues(self):
        """
        initialiseValues inital the values array
        """
        self.nodeDeltas = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def train(self, trainingSets, log=True, log_filename="train.log"):
        """train the mlp-network.
        
        Training of the mlp-network for a given `trainingSets` for maximum number of epchos. 
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        log : bool, optional
            log the current progress with time-stamp, epochs, and global error, by default True
        log_filename : str, optional
            name of the logging-file for the training, by default 'train.log'

        Returns
        -------
         : bool
            Return a bool for indicating successful (True) or failed (False) learning.
        """

        self.numEpochs = 1
        if log:
            logging.getLogger("train")
            logging.basicConfig(
                level=logging.DEBUG,
                filename="tmp.log",
                format="%(asctime)s\t%(levelname)-8s\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # logging.addLevelName(30, "WARNING")
            logging.addLevelName(32, "INFO:")
            logging.addLevelName(33, "OPTIMIZATION:")
            logging.addLevelName(34, "SUCCEED:")
            logging.addLevelName(35, "FAILED:")
            logging.log(32, "Number-of-Epochs\tGlobal-Error")
            # logging.info("\t----------------\t------------")
        # Have to change to a for-if slope
        while True:
            if self.numEpochs > self.maxNumEpochs:
                if log:
                    # Training failed
                    logging.log(35, "FAILED-AFTER:\t{}".format(self.numEpochs - 1))
                    logging.shutdown()
                    self.move_file(log_filename)
                return False
            sumNetworkError = 0
            for i, act_training_set in enumerate(trainingSets):
                # Switching to FeedForworad.py
                self.network.activate(act_training_set)
                # outputs = self.network.getOutputs()
                # Come back to Backpropagation.py
                self.calculateNodeDeltas(act_training_set)
                self.calculateGradients()
                self.calculateWeightUpdates()
                self.applyWeightChanges()
                sumNetworkError += self.calculateNetworkError(act_training_set)
            globalError = sumNetworkError / len(trainingSets)
            logging.log(33, "{}\t{}".format(self.numEpochs, globalError))
            self.error = globalError
            self.numEpochs = self.numEpochs + 1
            if globalError < self.minimumError:
                break
        if log:
            # Training suceed
            logging.log(34, "FINISHED-AFTER:\t{}".format(self.numEpochs - 1))
            logging.shutdown()
            self.move_file(log_filename)
            # logging.stream.close()
        return True

    def test(self, testSets, log=True, log_filename="test.log"):
        """train and verification of the mlp-network.
        
        Testing of the mlp-network for a given `testSets`. 
        
        Parameters
        ----------
        testSets : array
            The test set is provided as float-array where X- and y-values are keeped together.
        log : bool, optional
            print the current progress with global error, by default True
        log_filename : str, optional
            name of the logging-file for the test, by default 'train.log'

        Returns
        -------
         : bool
            Return a bool for indicating successful (True) or failed (False) prediction.
        """

        self.numEpochs = 1
        if log:

            logging.getLogger("test")
            logging.basicConfig(
                level=logging.DEBUG,
                filename="tmp.log",
                # handlers=[logging.FileHandler(log_filename), logging.getLogger(log_filename)],
                format="%(asctime)s\t%(levelname)-8s\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # logging.addLevelName(30, "WARNING")
            logging.addLevelName(32, "INFO:")
            logging.addLevelName(33, "OPTIMIZATION:")
            logging.addLevelName(34, "SUCCEED:")
            logging.addLevelName(35, "FAILED:")
            logging.log(32, "Reference-Values\tPredicted-Values\tMSE-Error")
            # logging.info("\t----------------\t------------")
        # Have to change to a for-if slope

        sumNetworkError = 0
        for i, act_testSet in enumerate(testSets):
            # Switching to FeedForworad.py
            self.network.activate(act_testSet)
            # outputs = self.network.getOutputs()
            # Come back to Backpropagation.py
            self.calculateNodeDeltas(act_testSet)
            sumNetworkError, ref_out, pre_out = self.calculateNetworkError(
                act_testSet, verbose=True
            )
            print(ref_out, pre_out, sumNetworkError)
            # globalError = sumNetworkError / len(testSets)
            # print(self.network.activate(testSets))
            # if globalError < self.minimumError:
            #    return True
            # else:
            #    return False
            logging.log(33, "{}\t{}\t{}".format(ref_out, pre_out, sumNetworkError))

        # if log:
        # Training suceed
        #    logging.log(34,"FINISHED-AFTER:\t{}".format(self.numEpochs - 1))
        logging.shutdown()
        self.move_file(log_filename)
        # return True

    def calculateNodeDeltas(self, trainingSet):
        """calculateNodeDeltas, error of each node.
        
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        """
        referenceOutputs = trainingSet[
            -1 * self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        ]
        # Initial phase

        actl_node = [
            self.networkLayers[len(self.networkLayers) - 1]["start_node"],
            self.networkLayers[len(self.networkLayers) - 1]["end_node"] + 1,
        ]
        activation = self.network.getActivation()
        error = self.network.values[actl_node[0] : actl_node[1]] - referenceOutputs

        self.nodeDeltas[actl_node[0] : actl_node[1]] = np.multiply(
            -error,
            activation.getDerivative(self.network.net[actl_node[0] : actl_node[1]]),
            dtype=self.dtype,
        )

        for k in range(len(self.networkLayers) - 2, 0, -1):

            actl_node = [
                self.networkLayers[k]["start_node"],
                self.networkLayers[k]["end_node"] + 1,
            ]
            connectNode = len(self.network.getWeight())
            # Calculating the node deltas
            self.nodeDeltas[actl_node[0] : actl_node[1]] = np.multiply(
                np.dot(
                    self.network.weights[actl_node[0] : actl_node[1]],
                    self.nodeDeltas[:connectNode],
                ),
                activation.getDerivative(self.network.net[actl_node[0] : actl_node[1]]),
                dtype=self.dtype,
            )

    def calculateGradients(self):
        """calculateGradients, gradient of each value and bias.
        """

        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            # Value-Gradient
            self.gradients[
                prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
            ] = np.outer(
                self.network.values[prev_index[0] : prev_index[1]],
                self.nodeDeltas[actl_index[0] : actl_index[1]],
                # dtype=self.dtype,
            )
            # Bias-Gradient
            self.biasGradients[num, actl_index[0] : actl_index[1]] = self.nodeDeltas[
                actl_index[0] : actl_index[1]
            ]

    def calculateWeightUpdates(self):
        """calculateWeightUpdates of the 'new' weights and bias-weights.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            # Updating the weights
            self.weightUpdates[
                prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
            ] = np.add(
                np.multiply(
                    self.learningRate,
                    self.gradients[
                        prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                    ],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta,
                    self.weightUpdates[
                        prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                    ],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )
            # Updating the bias-weights
            self.biasWeightUpdates[num, actl_index[0] : actl_index[1]] = np.add(
                np.multiply(
                    self.learningRate,
                    self.biasGradients[num, actl_index[0] : actl_index[1]],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta,
                    self.biasWeightUpdates[num, actl_index[0] : actl_index[1]],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )

    def applyWeightChanges(self):
        """applyWeightChanges of the gradient correction to the layers.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            self.network.updateWeight(
                prev_index,
                actl_index,
                self.weightUpdates[
                    prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                ],
            )
            self.network.updateBiasWeight(
                num,
                actl_index,
                self.biasWeightUpdates[num, actl_index[0] : actl_index[1]],
            )

    def calculateNetworkError(self, trainingSet, verbose=False):
        """calculateNetworkError based on the the mean squared error.
        
        
        calculateNetworkError is using the mean squared error (MSE) for measuring the average of the squares of the errors. 
        In this context, the average squared difference between the predicted values and the real values (training set).
        
        Parameters
        ----------
        trainingSet : array
            The training-set with X,y for validation of the optimization-cycle
        verbose : bool, optional
            Activate the returns of predicted and real output-values, by default False
        
        Returns
        -------
        globalError : float
            Global Error as a non-negative floating point value (the best value is 0.0); defined as MSE
        referenceOutputs : array or float, optional
            Reference y-values (normally float, but can be an array for multi-output-learning)
        predictedOutputs : array or float, optional
            Predicted y-values (normally float, but can be an array for multi-output-learning)
        """
        # Getting the y-values from back to the top
        referenceOutputs = trainingSet[-1 * self.networkLayers[-1]["num_nodes"] :]
        # Very inperformant this array-cut, has to be tuned
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        numNodes = self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        predictedOutputs = self.network.values[startNode : endNode + 1]
        globalError = np.mean(
            np.square(
                np.subtract(referenceOutputs, predictedOutputs, dtype=self.dtype,),
                dtype=self.dtype,
            ),
            dtype=self.dtype,
        )
        if verbose:
            return globalError, referenceOutputs, predictedOutputs
        else:
            return globalError

    def getGlobalError(self):
        """Returns the global error.
        
        Returns
        -------
        error : float
            MSE-based global error
        """
        return self.error

    @staticmethod
    def move_file(log_filename):
        try:
            os.rename("tmp.log", log_filename)
        except FileExistsError:
            os.replace("tmp.log", log_filename)


if __name__ == "__main__":

    # Ssigmoid = Sigmoid()

    networkLayer = [2, 5, 1]

    # feedForward = FeedForward(networkLayer, "Sigmoid")

    backpropagation = Backpropagation(
        networkLayer, "Sigmoid", 0.7, 0.5, maxNumEpochs=100000
    )

    trainingSet = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    backpropagation.initialise()
    result = backpropagation.train(trainingSet)
    backpropagation.test(trainingSet)

    # feedForward.save('./networkLayer.txt')
