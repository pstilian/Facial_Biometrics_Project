#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:00:14 2021

@author: kaptain
"""

import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('image_0228.jpg',1)
print("Image Properties")
print("- Number of Pixels: " + str(image.size))
print("- Shape/Dimensions: " + str(image.shape))
#b,g,r = cv2.split(image)			#分别提取B、G、R通道
#img1 = cv2.merge([r,g,b])	#重新组合为R、G、B
#plt.figure()
#plt.title("img1")
#plt.imshow(img1)

def rgb2gray(img):
    b=img[:,:,0].copy()
    g=img[:,:,1].copy()
    r=img[:,:,2].copy()
 
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    img[:,:,0]=out
    img[:,:,1]=out
    img[:,:,2]=out
    img = img.astype(np.uint8)
    
    return img


#img_gs = rgb2gray(image)

#plt.figure()
#plt.title("gray")
#plt.imshow(img_gs)





#img_gs = cv2.imread('image_0228.jpg', cv2.IMREAD_GRAYSCALE) # Convert image to grayscale
# RGB、HSV、YCrCb、XYZ、Lab
img_g = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure()
plt.title("original")
plt.imshow(img_g)

img_g2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.figure()
plt.title("HSV")
plt.imshow(img_g2)

img_g3 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
plt.figure()
plt.title("YCrCb")
plt.imshow(img_g3)

img_g4 = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
plt.figure()
plt.title("XYZ")
plt.imshow(img_g4)

img_g5 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
plt.figure()
plt.title("Lab")
plt.imshow(img_g5)

img_gs = cv2.cvtColor(img_g, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.title("gray")
plt.imshow(img_gs,cmap='gray')

# Adding salt & pepper noise to an image
def salt_pepper(prob):
      # Extract image dimensions
      row, col = img_gs.shape

      # Declare salt & pepper noise ratio
      s_vs_p = 0.5
      output = np.copy(img_gs)

      # Apply salt noise on each pixel individually
      num_salt = np.ceil(prob * img_gs.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img_gs.shape]
      output[coords] = 1

      # Apply pepper noise on each pixel individually
      num_pepper = np.ceil(prob * img_gs.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img_gs.shape]
      output[coords] = 0
      #cv2_imshow(output)
      plt.figure()
      plt.title("output")
      plt.imshow(output,cmap='gray')

      return output

# Call salt & pepper function with probability = 0.5
# on the grayscale image of rose
sp_05 = salt_pepper(0.05)

# Store the resultant image as 'sp_05.jpg'
#cv2.imwrite('sp_05.jpg', sp_05)




# Create our sharpening kernel, the sum of all values must equal to one for uniformity
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

# Applying the sharpening kernel to the grayscale image & displaying it.
print("\n\n--- Effects on S&P Noise Image with Probability 0.5 ---\n\n")

# Applying filter on image with salt & pepper noise
sharpened_img = cv2.filter2D(sp_05, -1, kernel_sharpening)

plt.figure()
plt.title("sharpened_img")
plt.imshow(sharpened_img,cmap='gray')

