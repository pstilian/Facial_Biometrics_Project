# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:20:26 2021

@author: 30510
"""

import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("image_0228.jpg", 1)
h, w, c = img1.shape
print("Dimensions of the image is:nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c)
print(type(img1))
print(img1.dtype)
print(img1)
#cv2.imshow('image_0228.jpg', img)
b,g,r = cv2.split(img1)			#分别提取B、G、R通道
img = cv2.merge([r,g,b])	#重新组合为R、G、B
plt.figure()
plt.title("img")
plt.imshow(img)

#k = cv2.waitKey(0)
#if k == 27 or k == ord('q'):
    #cv2.destroyAllWindows()
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Mandrill_grey.jpg', gray)
plt.figure()
plt.title("gray")
plt.imshow(gray,cmap='gray')

import matplotlib.pyplot as plt
import numpy as np

def extract_bit_plane(cd):
    #  extracting all bit one by one 
    # from 1st to 8th in variable 
    # from c1 to c8 respectively 
    c1 = np.mod(cd, 2)
    c2 = np.mod(np.floor(cd/2), 2)
    c3 = np.mod(np.floor(cd/4), 2)
    c4 = np.mod(np.floor(cd/8), 2)
    c5 = np.mod(np.floor(cd/16), 2)
    c6 = np.mod(np.floor(cd/32), 2)
    c7 = np.mod(np.floor(cd/64), 2)
    c8 = np.mod(np.floor(cd/128), 2)
    # combining image again to form equivalent to original grayscale image 
    cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
    to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()
    return cc

reconstructed_image = extract_bit_plane(gray)