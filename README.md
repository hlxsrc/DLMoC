# DLMoC

DLMoC or Deep Learning Model Creator is a project aimed to train and create deep learning models, print useful information of the training and plot results. 

# Project Structure

```
|-- DLMoC/
|   |-- __init__.py
|   |-- datasets/
|   |   |-- __init__.py
|   |   |-- simpledatasetloader.py
|   |-- nn/
|   |   |-- __init__.py
|   |   |-- conv/
|   |   |   |-- __init__.py
|   |   |   |-- shallownet.py
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   |-- imagetoarraypreprocessor.py
|   |   |-- simplepreprocessor.py
|   |-- shallownet_load.py
|   |-- shallownet_train.py
```

- DLMoC/
- datasets/
  - `simpledatasetloader.py` is used to load small image datasets from disk, preprocessing and return images and class labels.
- nn/
  - conv/
    - `shallownet.py` is used to create the ShallowNet architecture. 
- preprocessing/
  - `imagetoarraypreprocessor.py` is used to accept an input image and properly order channels based on defined settings.
  - `simplepreprocessor.py` is used to load an image from disk and resize it to a fixed size ignoring aspect ratio. 
- `shallownet_load.py` is used to load a pre-trained model from disk and label images with predictions. 
- `shallownet_train.py` is used to train and create the model, receives images as input and creates hdf5 model as output. 

# Usage

To train and create a model with `shallownet_train.py`

```
python shallownet_train.py --dataset /path/to/dataset \
  --model shallownet_weights.hdf5
```

To load a pre-trained model from disk with `shallownet_load.py`

```
python shallownet_load.py --dataset /path/to/dataset \
  --model shallownet_weights.hdf5
```

# Architectures

## ShallowNet

Simple architecture. Only contains a few layers. The architecture can be summarized as:

` INPUT => CONV => RELU => FC `

# Acknowledgments

This deep learning model creator is based on the examples created by Adrian Rosebrock on the [Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) book from PyImageSearch.
