# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:00:20 2021

@author: 30510
"""


import cv2
from insightface.model_zoo import face_detection

# 读取文件
image = cv2.imread('1.jpg')  # 读取图片
model = face_detection.retinaface_r50_v1()  # 加载模型
model.prepare(ctx_id=-1, nms=0.4)

# 人脸检测
faces, landmark = model.detect(image, threshold=0.5, scale=1.0)
for face in faces:
    (x, y, w, h, confidence) = face
    pt1 = int(x), int(y)
    pt2 = int(w), int(h)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness=2)  # 画出人脸矩形框

    text = '{:.2f}%'.format(confidence * 100)  # 置信度文本
    startX, startY = pt1
    y = startY - 10 if startY - 10 > 10 else startY + 10
    org = (startX, y)  # 文本的左下角坐标
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)  # 画出置信度

# 显示和保存图片
cv2.imshow('result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg', image)
print('已保存')
