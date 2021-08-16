import numpy as np
from numpy.core.fromnumeric import argmax

class NeuralNetwork:
    def __init__(self, layerSize):
        #   A list of weight matrices dimensions based on the layerSize tuple
        #   Number of weight Matrices will be layerSize's lenght - 1
        weightMatricesDimensions = [(i, j) for i, j in zip(layerSize[1:],  layerSize[:-1])]

        #   The weight matrices filled with zeros
        #   The first argument taken by np.zeros() is the "shape" 
        #   In mathimatical words... "dimension", hence dims
        #   weightArrays = [np.zeros(dims) for dims in weightMatricesDimensions]
        self.weightMatrices = [np.random.standard_normal(dims)/np.sqrt(layerSize[0]) for dims in weightMatricesDimensions]

        #   Creating biases for each layer except the input
        self.biases = [np.zeros((layerLength, 1)) for layerLength in layerSize[1:]]

    def predict(self, a):
        for w, b in zip(self.weightMatrices, self.biases):
            a=self.activationSigmoid(np.matmul(w, a) + b)
        return a

    def accuracy(self, images, lables):
        predictions = self.predict(images)
        correctCount = sum([np.argmax(i) == np.argmax(j) for i, j in zip(predictions, lables)])
        print("{0}/{1} accuracy: {2}%".format(correctCount, len(lables), (correctCount/len(lables))*100))

    @staticmethod
    def activationSigmoid(x):
        return 1/(1+np.exp(-x))