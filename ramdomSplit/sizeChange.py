# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 00:05:31 2021

@author: 30510
"""
import cv2
import os
import sys


def resizeImage2(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)

    w = image.shape[1]    
    if w == 250:  
        return
    else:
        resizeImg = cv2.resize(image,(250,250))
        cv2.imwrite(file,resizeImg)
        cv2.waitKey(100)

def resizeAll2(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:      
        file = str(file)
        resizeImage2(file,NoResize)     
    os.chdir(currentpath)    
    sys.stdin.flush()       


def main2(PATH):
    for childPATH in os.listdir(PATH): 
        childPATH = PATH + '/'+ str(childPATH)    
        resizeAll2(childPATH)


    return