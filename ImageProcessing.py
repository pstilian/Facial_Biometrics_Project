# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:24:39 2021

@author: Peter Stilian
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
This function takes in a filepath of an image and converts it to grayscale version 2 will simply take in an array
of input images an output an array of grayscale images
Parmeters: image, The filepath for the image to be downloaded  Returns: gray_image a grayscale version of original image
"""
def Grayscale(image):
    img = cv2.imread(image, 0)
    #cv2.imshow('Original Image', im)
    #cv2.waitKey(0)
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('Grayscale Image', gray_image)
    #cv2.waitKey(0)
    
    #cv2.destroyAllWIndows()

    return gray_image    

"""
"""
def Canny_Edge(image):
    img = cv2.imread(image, 0)
    edges = cv2.Canny(img, 100, 200)  # 2nd and 3rd parameters are lower and higer intensity gradients
    
    #cv2.imshow('Canny Image', edges)
    #cv2.waitkey(0)
    
    return edges

"""
"""
def Gauss_Blur(image):
    img = cv2.imread(image, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0) # 2nd parameter is responsible for blurring kernel greater size = greater blur
    
    #cv2.imshow('Blurred Image', blur)
    #cv2.waitKey(0)
    
    return blur