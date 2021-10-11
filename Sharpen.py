#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:41:07 2021

@author: kaptain
"""
from matplotlib import pyplot as plt
import numpy as np
import cv2

def grey_avg2(image):
    
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            r,g,b = img_rgb[y,x]
            r = np.uint8(r * 0.299)
            g = np.uint8(g * 0.587)
            b = np.uint8(b * 0.114)

            rgb = np.uint8(r * 0.299 + b * 0.114 + g * 0.587)
            dist[y,x] = rgb
    return dist


def grey_avg(image):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            avg = sum(image[y,x]) / 3
            dist[y,x] = np.uint8(avg)
    
    return dist

src = cv2.imread('A00011.jpg')

avg1 = grey_avg(src)
avg2 = grey_avg2(src)

cv2.imshow('src',src)
cv2.imshow('avg-1',avg1)
cv2.imshow('avg-2',avg2)

cv2.waitKey(0)
cv2.destroyAllWindows()

img_gs = avg1

# Create our sharpening kernel, the sum of all values must equal to one for uniformity
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

# Applying the sharpening kernel to the grayscale image & displaying it.
print("\n\n--- Effects on S&P Noise Image with Probability 0.5 ---\n\n")

# Applying filter on image with salt & pepper noise
sharpened_img = cv2.filter2D(img_gs, -1, kernel_sharpening)

plt.figure()
plt.title("sharpened_img")
plt.imshow(sharpened_img)