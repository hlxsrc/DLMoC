# import necessary packages
import numpy as np
import cv2
import os

# define class
class SimpleDatasetLoader:
    
    # define constrcutori
    # receives: list of image preprocessors
    def __init__(self, preprocessors=None):
        
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None
        # initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    # load method
    # receives: imagePaths list of paths to images
    def load(self, imagePaths, verbose=-1):
        
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            
            # load the image and extract the class label assuming
            # that our path has the following format: 
            # /path/to/dataset/<class>/<image>.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if preprocessors are not None
            if self.preprocessors is not None:
                
                # loop over the preprocessors and
                # apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat processed image as a "feature vector" 
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label) 

            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, 
                    len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))


