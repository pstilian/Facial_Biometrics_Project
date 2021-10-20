# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:24:39 2021

@author: Peter Stilian
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

"""
This function takes in a filepath of an image and converts it to grayscale version 2 will simply take in an array
of input images an output an array of grayscale images
Parmeters: image, The filepath for the image to be downloaded  Returns: gray_image a grayscale version of original image
"""
def Grayscale(image):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    cv2.imshow('Original Image', img)
    #cv2.waitKey(0)
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('Grayscale Image', gray_image)
    #cv2.waitKey(0)
    
    #cv2.destroyAllWIndows()
    
    #cv2.imwrite('E:\gray.png',gray_image)

    return gray_image    



def Canny_Edge(image):
    img = cv2.imread(image, 0)
    edges = cv2.Canny(img, 80, 110)  # 2nd and 3rd parameters are lower and higer intensity gradients
    edges2 = cv2.Canny(img, 30, 150)
    #cv2.imshow('Canny Image', edges)
    #cv2.waitKey(0)
    
    #cv2.imwrite('E:\canny.png',edges)
    
    return edges, edges2



def Gauss_Blur(image):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    blur = cv2.GaussianBlur(img, (55, 55), 0) # 2nd parameter is responsible for blurring kernel greater size = greater blur
    
    #cv2.imshow('Blurred Image', blur)
    #cv2.waitKey(0)
    
    #cv2.imwrite('E:\blur.png',blur)
    
    return blur

g = Grayscale('E:\Project 1 Database\AntonioLaverghetta/BL_BL_2.png')
cv2.imwrite('E:\gray.png',g)
    
c1, c2 = Canny_Edge('E:\Project 1 Database\AntonioLaverghetta/BL_BL_2.png')
cv2.imwrite('E:\canny.png',c1)
cv2.imwrite('E:\canny2.png',c2)

ga = Gauss_Blur('E:\Project 1 Database\AntonioLaverghetta/BL_BL_2.png')
cv2.imwrite('E:\gauss.png',ga)
