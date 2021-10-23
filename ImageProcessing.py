# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:24:39 2021

@author: Peter Stilian

This program is designed to be ran inside it's current directory with the dataset located in the parent directory
and named 'Project 1 Database' If a different Database folder is to be used rename the parameter for get_images (Line 53).

Returns: Fully converted datasets in Grayscale, Canny Edge detected and gaussian Blurred forms
"""

import cv2
import os


def get_images(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV                    
                    img = cv2.imread(
                            os.path.join(image_directory, subfolder, file)
                            )
                    # resize the image                     
                    width = 224
                    height = 224
                    dim = (width, height)
                    img = cv2.resize(img, dim)
                    # add the resized image to a list X
                    X.append(img)
                    
                    # add the image's label to a list y
                    y.append(subfolder)
    
    print("All images are loaded")     
    # return the images and their labels      
    return X, y

x, y = get_images('../Project 1 Database')

gray_dir = '../GrayscaleDatabase/'
canny_dir = '../CannyDatabase/'
blur_dir = '../BlurDatabase/'


for i in range(len(x)):
    curimg = x[i]
    curdir = y[i]
    
    path = os.path.join(gray_dir, curdir)
    os.makedirs(path, exist_ok=True)   
    g = cv2.cvtColor(curimg, cv2.COLOR_BGR2GRAY)    
    filename = path + '/' + curdir + '_' + str(i) + '_grayscale.png'
    cv2.imwrite(filename, g)
    
    path = os.path.join(canny_dir, curdir)
    os.makedirs(path, exist_ok=True) 
    temp = cv2.GaussianBlur(g, (3, 3), 0)
    c = cv2.Canny(temp, 30, 150)   
    filename = path + '/' + curdir + '_' + str(i) + '_canny.png'
    cv2.imwrite(filename, c)
    
    path = os.path.join(blur_dir, curdir)
    os.makedirs(path, exist_ok=True)   
    b = cv2.GaussianBlur(curimg, (15, 15), 0)    
    filename = path + '/' + curdir + '_' + str(i) + '_blur.png'
    cv2.imwrite(filename, b)
    
print('Database Creation Complete')