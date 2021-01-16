# import the necessary packages

# project libraries
from DLMoC.preprocessing import ImageToArrayPreprocessor
from DLMoC.preprocessing import SimplePreprocessor
from DLMoC.datasets import SimpleDatasetLoader

# keras library
# used to load trained model from disk
from keras.models import load_model

# aux libraries
from imutils import paths
import numpy as np
import argparse
import cv2

# import from tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# fix problems with tensorflow
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
        help="path to input dataset")
ap.add_argument("-m", "--model", required=True, 
        help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["face", "body", "other"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
# resize input images to 32x32
sp = SimplePreprocessor(32, 32)
# handle channel ordering
iap = ImageToArrayPreprocessor()
# this is important to recognize patterns

# supply list of preprocessors (sequentially)
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
# load dataset from disk
(data, labels) = sdl.load(imagePaths)
# scale raw pixel intensities in range [0, 1]
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
