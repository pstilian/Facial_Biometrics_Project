# Facial_Biometrics_Project

This facial biometric system has two major components: data preprocessing/augmentation and training/testing models.

Before you continue with running the system, please ensure that you have a dataset of labeled images with the appropriate biometric features to be tested. Each subject in the dataset should have its own directory with its corresponding images placed inside.

ImageProcessing.py:
The data preprocessing is handled in ImageProcessing.py and will apply grayscale, Canny edge detection, and blur to the images. This program accepts images of .jpg, .png, and .gif file types, and will go through each directory in the dataset to apply all three forms of data preprocessing. The preprocessing on the dataset will be stored separately so that the original images are not be modified. Line 55 expects the location of the dataset to be preprocessed, and lines 57 - 59 expect three different directories to store the newly preprocessed data.

data_augmentation.py:
The data augmentation is handled in data_augmentation.py and will slice each image in a dataset into a certain number of square tiles. The number of tiles could only be a perfect square, and each square tile was equal in pixel length. The slicing of the images began from the top left corner of the image and moved across the columns of the image to the right and down each row. This program does not modify the original dataset and stores the newly generated dataset in a different directory. Line 11 expects the location of the original dataset, and line 12 expects the output directory for the augmented dataset.

main.py:
Contains the main training loop for the neural models. Execute this file to train the FCC model.

models.py:
Contains the implementation of each of our neural models.

performance_plots.py:
Various helper functions for graphing performance, identical to the guided learning activity.

get_images.py:
Reads through the images of a dataset, resizes the images, and returns a pandas dataframe storing the images.

Svm.py:
Contains the SVM implementation of our facial biometric system.

sizeChange.py:
Contains a few functions that resizes the images of a dataset.

FaceCapture.py：
Contains a few functions that automatically identify a bounding box around the face, and crop anything outside it.

data_reader.py
Reads through the images of a dataset.

feature_extractor.py
contains the class FeatureExtractor. This class implements the fit-transform paradigm. Given a list of images, it computes the eigenfaces and saves them for future use.

image.py
contains the Image class, which is a class that wraps a 2D numpy array and creates an abstraction for an image.

training.py: 
Contains the 10 fold corss validation and several classifiers to train our facial biometric system.

main.py: 
recognizes the face in a new image using the model learned by training.py.



### Required Packages:

1. OpenCV
2. PyTorch
3. Pandas
4. Transformers
5. dlib
6. Scikit-Learn
