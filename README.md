# RTHuD

Real-Time Human Detector

# Project Structure

```
|-- RTHuD/
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
```

- RTHuD is the main directory
- datasets/
  - simpledatasetloader.py: is used to load small image datasets from disk, preprocessing and return images and class labels.
- nn/
  - conv/
    - shallownet.py: is used to create the ShallowNet architecture. 
- preprocessing/
  - imagetoarraypreprocessor.py: is used to accept an input image and properly order channels based on defined settings.
  - simplepreprocessor.py: is used to load an image from disk and resize it to a fixed size ignoring aspect ratio. 

# Architectures

## ShallowNet

Simple architecture. Only contains a few layers. The architecture can be summarized as:

` INPUT => CONV => RELU => FC `

# Acknowledgments

This real-time human detector is based on the examples created by Adrian Rosebrock on the [Deep Learning for Computer Vision with Python. 1, Starter Blunde](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) book from PyImageSearch.
