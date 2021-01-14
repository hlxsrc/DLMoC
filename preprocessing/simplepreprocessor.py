# import the necessary packages 
import cv2

# define class
class SimplePreprocessor:

    # define constructor
    # receives: width, height and optionally algorithm for
    #   resizing
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    
    # preprocess function 
    # receives: input image
    def preprocess(self, image):
        # resize the image to a fixed size
        # ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height),
                interpolation=self.inter)


