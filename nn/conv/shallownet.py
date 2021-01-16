# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k

# define class
class ShallowNet:

    # build method
    # receives: 
    #   width - width of input images (columns)
    #   height - height of input images (rows)
    #   depth - channels of input image
    #   classes - number of classes to predict
    @staticmethod
    def build(width, height, depth, classes):
        
        # initialize the model
        # initalize model with shape "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if using "channels first"
        # update input shape
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define CONV => RELU layer
        # convolutional layer with 32 filters
        # each filter of 3x3
        # same padding to ensure:
        #   size of output = size of input
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        # apply ReLU activation 
        model.add(Activation("relu"))

        # flatten multi-dimensional representation into 1D list
        model.add(Flatten())
        # dense layer w/ same number of nodes as output class labels
        model.add(Dense(classes))
        # apply softmax activation function for class probabilities
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model




