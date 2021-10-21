# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:47:08 2021

@author: 30510
"""
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"

# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import os
import re
import sys
from PIL import Image
import string
import numpy as np

"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
PATH = 'ada'   #这里路径自己定,注意要是直接从我的电脑复制路径要加 r'****'
#我这里是相对路径,亲测中文路径也可以

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def resizeImage(file,NoResize,i):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)
    #i = 0
    #如果type(image) == 'NoneType',会报错,导致程序中断,所以这里先跳过这些图片,
    #并记录下来,结束程序后手动修改(删除)
    
    if image is None:
        NoResize += [str(file)]
    else:
        
        rects=detect(image,cascade)
        for x1,y1,x2,y2 in rects:
        #调整人脸截取的大小。横向为x,纵向为y
            image = image[y1+10 :y2+20, x1+10 :x2 ]   
            resizeImg = cv2.resize(image,(92,112))
            cv2.imwrite(file,resizeImg)
            cv2.waitKey(100)
            grayImage = cv2.imread(os.path.realpath(file), cv2.IMREAD_GRAYSCALE)
        #iowrite = pgm_path.strip('\u202a') +'\%d.pgm'%i
        #i += 1
            cv2.imwrite(os.path.realpath(file).strip(get_file_name(file) + ".png") + str(i)+ ".pgm", grayImage)
            cv2.waitKey(100)
        #Image.open(os.path.realpath(file)).convert('L').save(pgm_path.strip('\u202a') +'\%d.pgm'%i)
        #i += 1


def get_file_name(path_string):
    """获取文件名称"""
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(path_string)
    if data:
        return data[0]



def resizeAll(root):
    
    #待修改文件夹
    fileList = os.listdir(root)
    #print("fileList")
    #print(fileList)
    #print (len(fileList))
    #输出文件夹中包含的文件        
    #print("修改前："+str(fileList))
    #得到进程当前工作目录
    currentpath = os.getcwd()   
    #将当前工作目录修改为待修改文件夹的位置    
    os.chdir(root)
    
    NoResize = []  #记录没被修改的图片
    i = 1
    for file in fileList:       #遍历文件夹中所有文件
        
        #print("file:")
        #print(file)
        # Without this code, the path will be wrong
        file = str(file)
        #print(file)
        #print("real path of file:")
        #print(os.path.realpath(file))
        resizeImage(file,NoResize,i)
        i += 1
         
    print("---------------------------------------------------")
    
    os.chdir(currentpath)       #改回程序运行前的工作目录
    
    sys.stdin.flush()       #刷新
    

    
    
    print('没别修改的图片: ',NoResize)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
pgm_path = '‪F:\\MobileBiometrics\\GuideLearning2\\adc'
size_m = 92
size_n = 112
if __name__=="__main__":
    #子文件夹
    for childPATH in os.listdir(PATH): #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        #子文件夹路径
        print("childPATH:")
        print(childPATH)
        childPATH = PATH + '/'+ str(childPATH)
        #print(PATH)
        print(childPATH)
        resizeAll(childPATH)
    print('------修改图片大小全部完成-_-')
    

