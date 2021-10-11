# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:47:08 2021

@author: 30510
"""


# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import os
import re
import sys
from PIL import Image
import string
import numpy as np

PATH = 'Caltech Faces Dataset'   #这里路径自己定,注意要是直接从我的电脑复制路径要加 r'****'
#我这里是相对路径,亲测中文路径也可以


def resizeImage(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)
    
    #如果type(image) == 'NoneType',会报错,导致程序中断,所以这里先跳过这些图片,
    #并记录下来,结束程序后手动修改(删除)
    
    if image is None:
        NoResize += [str(file)]
    else:
        resizeImg = cv2.resize(image,(250,250))
        cv2.imwrite(file,resizeImg)
        cv2.waitKey(100)

def resizeAll(root):
    
    #待修改文件夹
    fileList = os.listdir(root)
    #输出文件夹中包含的文件        
    # print("修改前："+str(fileList))
    #得到进程当前工作目录
    currentpath = os.getcwd()   
    #将当前工作目录修改为待修改文件夹的位置    
    os.chdir(root)
    
    NoResize = []  #记录没被修改的图片
    
    for file in fileList:       #遍历文件夹中所有文件
        file = str(file)
        resizeImage(file,NoResize)
         
    print("---------------------------------------------------")
    
    os.chdir(currentpath)       #改回程序运行前的工作目录
    
    sys.stdin.flush()       #刷新
    

    
    
    print('没别修改的图片: ',NoResize)

if __name__=="__main__":
    #子文件夹
    for childPATH in os.listdir(PATH): #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        #子文件夹路径
        childPATH = PATH + '/'+ str(childPATH)
        # print(childPATH)
        resizeAll(childPATH)
    print('------修改图片大小全部完成-_-')
    

