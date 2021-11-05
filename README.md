# Facial_Biometrics_Project

This facial biometric system has two major components: data preprocessing/augmentation and training/testing models.

Before you continue with running the system, please ensure that you have a dataset of labeled images with the appopriate biometric features to be tested. Each subject in the dataset should have its own directory with its corresponding images placed inside.

ImageProcessing.py:
The data preprocessing is handled in ImageProcessing.py and will apply grayscale, Canny edge detection, and blur to the images. This program accepts images of .jpg, .png, and .gif file types, and will go through each directory in the dataset to apply all three forms of data preprocessing. The preprocessing on the dataset will be stored separately so that the original images are not be modified. Line 55 expects the location of the dataset being preprocessed, and lines 57 - 59 expect three different directories to store the newly preprocessed data.

data_augmentation.py:
The data augmentation is handled in data_augmentation.py and will slice each image in a dataset into a certain number of square tiles. The number of tiles could only be a perfect square, and each square tile was equal in pixel length. The slicing of the images began from the top left corner of the image and moved across the columns of the image to the right and down each row. This program does not modify the original dataset and stores the newly generated dataset in a different directory. Line 11 expects the location of the original dataset, and line 12 expects the output directory for the augmented dataset.

