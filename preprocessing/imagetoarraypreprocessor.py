# import the necessary packages 
from keras.preprocessing.image import img_to_array

# define class
class ImageToArrayPreprocessor:

    # define constructor
    # receives: data format 
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    # preprocess function
    # receives: input image
    def preprocess(self, image):
        # apply Keras utility to correctly rearrange
        # the image dimensions
        return img_to_array(image, data_format=self.dataFormat)



