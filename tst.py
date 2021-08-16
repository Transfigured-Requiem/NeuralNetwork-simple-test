
#import matplotlib.pyplot as plt
import NeuralNetwork as nn
import numpy as np
import os


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist.npz')
with np.load(path) as data: 
	
#from keras.datasets import mnist
#   loading the minst.nps file
#   Each picture is 28x28 (px), total of 784 px stored in a column vector
#   Each image has a correspondig label coloumn vector (10x1), which digit the data represents
#   with np.load("mnist.npz") as data:
    #print(data.files)
    training_images = data["training_images"]
    training_lables = data["training_labels"]
#   Display code
#   plt.imshow(training_images[0].reshape(28, 28), cmap='gray')
#   plt.show()



#   Tuple containing the number of nuorons for each layer
layerSize = (784, 5, 10)
#x = np.ones((layerSize[0], 1))

network = nn.NeuralNetwork(layerSize)
#prediction = network.predict(training_images)
#print(np.argmax(prediction[0]))
network.accuracy(training_images, training_lables)