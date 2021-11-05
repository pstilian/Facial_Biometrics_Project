# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:03:42 2021

@author: 30510
"""


import cv2
import os
import re
import sys
import dlib
from os.path import join, getsize
import numpy as np

"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"

'''The PATH below is data file's directory'''
PATH = 'F://MobileBiometrics//Project1//10FoldValidation//data' 

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def resizeImage2(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)

    w = image.shape[1]    
    if w == 92 or w == 250:  
        return
    else:
        resizeImg = cv2.resize(image,(250,250))
        cv2.imwrite(file,resizeImg)
        cv2.waitKey(100)
        

def resizeImage1(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)

    
    if image is None:
        NoResize += [str(file)]
    else:
        w = image.shape[1]      
        if w != 92:
            rects=detect(image,cascade)
            for x1,y1,x2,y2 in rects:

                image = image[y1+10 :y2+20, x1+10 :x2 ]   
                resizeImg = cv2.resize(image,(92,112))
                cv2.imwrite(file,resizeImg)
                cv2.waitKey(10)
            
                #grayImage = cv2.imread(os.path.realpath(file), cv2.IMREAD_GRAYSCALE)

                #cv2.imwrite(file, grayImage)
                #cv2.waitKey(10)

def resizeImage3(file,NoResize):
    
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)
    
    if image is None:
        NoResize += [str(file)]
    else:
        w = image.shape[1]       
        if w !=92:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = model(image, 1)
            for face in faces:
                face = face.rect
                (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
                image = image[y1+10 :y2+20, x1+10 :x2 ]   
                resizeImg = cv2.resize(image,(92,112))
                cv2.imwrite(file,resizeImg)
                cv2.waitKey(100)
            
                #grayImage = cv2.imread(os.path.realpath(file), cv2.IMREAD_GRAYSCALE)

                #cv2.imwrite(file, grayImage)
                #cv2.waitKey(100)
        
def resizeImage6(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)
    
    if image is None:
        NoResize += [str(file)]
    else:
        w = image.shape[1]      
        if w != 92:
            height, width, channel = image.shape
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) 
            net.setInput(blob)
            detections = net.forward() 
            faces = detections[0, 0]
            for face in faces:
                confidence = face[2]  # 置信度
                if confidence > 0.5:  # 置信度阈值设为0.5
                    box = face[3:7] * np.array([width, height, width, height])  # 人脸矩形框坐标
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3]) # 右下角坐标
                    image = image[y1+10 :y2+20, x1+10 :x2 ] 
                    resizeImg = cv2.resize(image,(92,112))
                    cv2.imwrite(file,resizeImg)
                    cv2.waitKey(100)
            
                    #grayImage = cv2.imread(os.path.realpath(file), cv2.IMREAD_GRAYSCALE)
                    #cv2.imwrite(file, grayImage)
                    #cv2.waitKey(100)
                
def resizeImage4(file,NoResize):
    
    image = cv2.imread(file,cv2.IMREAD_COLOR)

    if image is None:
        NoResize += [str(file)]
    else:
        w = image.shape[1]     
        if w != 92:  
            resizeImg = cv2.resize(image,(92,112))
            cv2.imwrite(file,resizeImg)
            cv2.waitKey(100)
            #grayImage = cv2.imread(os.path.realpath(file), cv2.IMREAD_GRAYSCALE)
            #cv2.imwrite(file, grayImage)
            #cv2.waitKey(100)
            

def resizeImage5(file,NoResize,i,fileList,root):
    image = cv2.imread(file)
  
    if image is None:
        NoResize += [str(file)]
    else:
        oldname=root+ os.sep + get_file_name(file) + '.png' 
        #newname=root + os.sep + get_file_name(file) + '.pgm'
        newname=root + os.sep + str(i + 1)+'.pgm'           
        os.rename(oldname,newname)

def rename(dir):
    subdir = os.listdir(dir) 
    j = 0
    for i in subdir:
        path = os.path.join(dir,i)   
        oldname=path 
        newname=PATH + os.sep + 's' + str(j+1)    
        j += 1       
        os.rename(oldname,newname)      


def get_file_name(path_string):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(path_string)
    if data:
        return data[0]

def resizeAll1(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:      
        file = str(file)
        if k < 4:
            resizeImage1(file,NoResize)
        else:
            print('cannot capture this face')
            global k 
            k = 1
            global q
            global label0
            label0[q] = 1
            break
    print('jump out of the loop') 
    global q
    global label0
    label0[q] = 1       
    print("---------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush() 


def resizeAll2(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:      
        file = str(file)
        resizeImage2(file,NoResize)     
    print("---------------------------------------------------")
    os.chdir(currentpath)    
    sys.stdin.flush()       

    
def resizeAll3(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:      
        file = str(file)
        if k < 4:
            resizeImage3(file,NoResize)
        else:
            print('cannot capture this face')
            global k 
            k = 1
            global h
            global label
            label[h] = 1
            break
    print('jump out of the loop')
    global h
    global label
    label[h] = 1            
    print("---------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush()


def resizeAll6(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:      
        file = str(file)
        if k < 6:
            resizeImage6(file,NoResize)
        else:
            print('cannot capture this face')
            global k 
            k = 1
            global u
            global label3
            label3[u] = 1
            break
    print('jump out of the loop')
    global u
    global label3
    label3[u] = 1            
    print("---------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush()


def resizeAll4(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    for file in fileList:       
        file = str(file)
        resizeImage4(file,NoResize)     
    print("---------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush()

    
def resizeAll5(root):
    fileList = os.listdir(root)
    currentpath = os.getcwd()   
    os.chdir(root)
    NoResize = [] 
    i = 0
    for file in fileList:       
        file = str(file)
        resizeImage5(file,NoResize,i,fileList,root)
        i += 1
    print("---------------------------------------------------")
    os.chdir(currentpath) 
    sys.stdin.flush() 
 
    
def getdirsize(dir):  
   size = 0.0  
   for root, dirs, files in os.walk(dir):  
      size += sum([getsize(join(root, name)) for name in files])  
   return size  

def main1():
    #if __name__=="__main__":
    global q
    q = 0
    for childPATH in os.listdir(PATH):
        childPATH = PATH + '/'+ str(childPATH) 
        fileList = os.listdir(PATH)
        print(childPATH)
        print(len(fileList))
        if len(label0) == 0:
            for i in range(len(fileList)):
                global label0
                label0.append(0)
        if label0[q] == 0:
            resizeAll1(childPATH)
            
        if label0[q] == 1:
            q += 1
        
    return
        
def main2():
    for childPATH in os.listdir(PATH): 
        childPATH = PATH + '/'+ str(childPATH)    
        resizeAll2(childPATH)
    print('------complete the size change -_-')

    return

def main3():

    #if __name__=="__main__":
    global h
    h = 0
    for childPATH in os.listdir(PATH): 
        childPATH = PATH + '/'+ str(childPATH)    
        fileList = os.listdir(PATH)
        print(childPATH)
        print(len(fileList))
        if len(label) == 0:
            for i in range(len(fileList)):
                global label
                label.append(0)
        if label[h] == 0:
            resizeAll3(childPATH)
            
        if label[h] == 1:
            h += 1
    return

def main6():

    #if __name__=="__main__":
    global u
    u = 0
    for childPATH in os.listdir(PATH): 
        childPATH = PATH + '/'+ str(childPATH)    
        fileList = os.listdir(PATH)
        print(childPATH)
        print(len(fileList))
        if len(label3) == 0:
            for i in range(len(fileList)):
                global label3
                label3.append(0)
        if label3[u] == 0:
            resizeAll6(childPATH)
            
        if label3[u] == 1:
            u += 1
    return

def main4():
    #if __name__=="__main__":
    for childPATH in os.listdir(PATH): 
        childPATH = PATH + '/'+ str(childPATH)    
        resizeAll4(childPATH)
    print('complete the color space change -_-')
    return

def main5():
    #if __name__=="__main__":
    for childPATH in os.listdir(PATH): #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        #子文件夹路径
        #print("childPATH:")  
        #print(childPATH) 
        childPATH = PATH + '/'+ str(childPATH)    
        #print(childPATH) 
        #print(os.path.getsize(childPATH)) 
        resizeAll5(childPATH)
    print('------complete the pic name change -_-')
    return


def retry1():
    try:
        main1()
        print('finish the loop1')
        return
    except Exception as e:
        print("another try")
        #main()
        global k
        k += 1
        retry1()
        
def retry2():
    try:
        main3()
        print('finish the loop2')
        return
    except Exception as e:
        print("another try")
        #main()
        global k
        k += 1
        retry2()

def retry3():
    try:
        main6()
        print('finish the loop3')
        return
    except Exception as e:
        print("another try")
        #main()
        global k
        k += 1
        retry3()
        
label = []
label0 = []
label3 = []
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel') 
model = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
k = 1
q = 0
h = 0
u = 0
'chang file name'
rename(PATH)

'chang file size'
main2()

'use method 3 to capture face'
print('method 3')
retry3()

retry2()

'use method 1 to capture face'
#print('method 1')
#retry1()

'change pictures, which were not captured, into gray color space'
main4()

'chang picture name'
main5()


    

