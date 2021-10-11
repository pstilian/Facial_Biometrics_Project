# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 20:52:38 2021

@author: 30510
"""


import chardet
import numpy as np
import cv2 as cv
import cv2
from PIL import Image
import sys
from matplotlib import pyplot as plt

def add_salt_noise(img, snr=0.5):
    # 指定信噪比
    SNR = snr
    # 获取总共像素个数
    size = img.size
    # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
    noiseSize = int(size * (1 - SNR))
    # 对这些点加噪声
    for k in range(0, noiseSize):
        # 随机获取 某个点
        xi = int(np.random.uniform(0, img.shape[1]))
        xj = int(np.random.uniform(0, img.shape[0]))
        # 增加噪声
        if img.ndim == 2:
            img[xj, xi] = 255
        elif img.ndim == 3:
            img[xj, xi] = 0
    return img

img=cv.imread("image_0228.jpg",1)
img_salt = add_salt_noise(img, snr=0.99)

blured = cv.blur(img, (3, 3))
blured1 =  cv.blur(img, (7,7))
blured2= cv.GaussianBlur(img, (3,3), 0)
blured3=cv.GaussianBlur(img, (7,7), 0)

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title(" meanblur")
plt.imshow(blured[:,:,::-1])
plt.subplot(122)
plt.imshow(blured1[:,:,::-1])

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title("GaussianBlur")
plt.imshow(blured2[:,:,::-1])
plt.subplot(122)
plt.imshow(blured3[:,:,::-1])

b1 = cv.medianBlur(img, 1)
b2 = cv.medianBlur(img, 9)
b3 = cv.bilateralFilter(img, 9, 5, 5)
b4 = cv.bilateralFilter(img, 9, 50, 50)


plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title("medianBlur")
plt.imshow(b1[:,:,::-1])
plt.subplot(122)
plt.imshow(b2[:,:,::-1])

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title("bilateralFilter")
plt.imshow(b3[:,:,::-1])
plt.subplot(122)
plt.imshow(b4[:,:,::-1])

c1 = cv2.boxFilter(img,-1,(2,2),normalize=False)
c2 = cv2.bilateralFilter(img,25,100,100)

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title("boxFilter")
plt.imshow(c1[:,:,::-1])
plt.subplot(122)
plt.title("bilateralFilter")
plt.imshow(c2[:,:,::-1])

kernel = np.ones((9,9),np.float32)/81
c3 = cv2.filter2D(img,-1,kernel)
b,g,r = cv2.split(img_salt)			#分别提取B、G、R通道
img_salt1 = cv2.merge([r,g,b])	#重新组合为R、G、B

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.title("original picture")
plt.imshow(img_salt1)
plt.subplot(122)
plt.title("cv2.filter2D")
plt.imshow(c3[:,:,::-1])


image = img_salt	

kernel_sharpen_1 = np.array([
	        [-1,-1,-1],
	        [-1,9,-1],
	        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
	        [1,1,1],
	        [1,-7,1],
	        [1,1,1]])
	
# #卷积
output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
	
plt.figure()
plt.title('Original Image')
plt.imshow(image[:,:,::-1])

plt.figure()
plt.title('sharpen_1 Image')
plt.imshow(output_1[:,:,::-1])

plt.figure()
plt.title('sharpen_2 Image')
plt.imshow(output_2[:,:,::-1])
plt.show()


