# RTHuD

Real-Time Human Detector

# Project Structure

```
|-- RTHuD
|   |-- __init__.py
|   |-- datasets
|   |   |-- __init__.py
|   |   |-- simpledatasetloader.py
|   |-- preprocessing
|   |   |-- __init__.py
|   |   |-- imagetoarraypreprocessor.py
|   |   |-- simplepreprocessor.py
```

- RTHuD is the main directory
- datasets 
  - simpledatasetloader.py is used to load small image datasets from disk, preprocessing and return images and class labels.
- preprocessing
  - imagetoarraypreprocessor.py is used to accept an input image and properly order channels based on defined settings.
  - simplepreprocessor.py is used to load an image from disk and resize it to a fixed size ignoring aspect ratio. 



