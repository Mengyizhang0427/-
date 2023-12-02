# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""

import streamlit as st   
import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode
import tempfile

st.balloons()

# Title
st.header("2D人脸表情识别系统")

# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images


#opencv自带的一个面部识别分类器
detection_model_path = 'haarcascade_frontalface_default.xml'

classification_model_path = 'model_DenseNet121_200Adam.pkl'

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)

# 加载表情识别模型
emotion_classifier = torch.load(classification_model_path)


frame_window = 10

#表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_window = []

if st.button("摄像头拍摄人脸表情识别（按q退出）"):
    # 调起摄像头，0是笔记本自带摄像头
    video_capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.startWindowThread()
    cv2.namedWindow('window_frame')

    while True:
        # 读取一帧
        _, frame = video_capture.read()
        frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
        frame = frame.copy()
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 获取当前帧中的全部人脸
        faces = face_detection.detectMultiScale(gray,1.3,5)
        # 对于所有发现的人脸
        for (x, y, w, h) in faces:
            # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
            cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

            # 获取人脸图像
            face = gray[y:y+h,x:x+w]

            try:
                # shape变为(48,48)
                face = cv2.resize(face,(48,48))
            except:
                continue

            # 扩充维度，shape变为(1,48,48,1)
            #将（1，48，48，1）转换成为(1,1,48,48)
            face = np.expand_dims(face,0)
            face = np.expand_dims(face,0)

            # 人脸数据归一化，将像素值从0-255映射到0-1之间
            face = preprocess_input(face)
            new_face=torch.from_numpy(face)
            new_new_face = new_face.float().requires_grad_(False)
            
            # 调用我们训练好的表情识别模型，预测分类
            emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
            emotion = emotion_labels[emotion_arg]

            emotion_window.append(emotion)

            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)

            try:
                # 获得出现次数最多的分类
                emotion_mode = mode(emotion_window)
            except:
                continue

            # 在矩形框上部，输出分类文字
            cv2.putText(frame,emotion_mode,(x,y-30), font, 1.0,(0,0,255),1,cv2.LINE_AA)

        try:
            # 将图片从内存中显示到屏幕上
            cv2.imshow('window_frame', frame)
        except:
            continue

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


    
uploaded_file = st.file_uploader("上传本地图片进行人脸表情识别", type=["jpg", "png", "jpeg"])
    
if uploaded_file is not None:
   # 使用OpenCV读取上传的图片文件
   file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
   image = cv2.imdecode(file_bytes, 1)
   frame = image
   font = cv2.FONT_HERSHEY_SIMPLEX
   # 获得灰度图，并且在内存中创建一个图像对象
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # 获取当前帧中的全部人脸
   faces = face_detection.detectMultiScale(gray,1.3,5)
   # 对于所有发现的人脸
   for (x, y, w, h) in faces:
        # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
        cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

        # 获取人脸图像
        face = gray[y:y+h,x:x+w]

        try:
            # shape变为(48,48)
            face = cv2.resize(face,(48,48))
        except:
            continue

        # 扩充维度，shape变为(1,48,48,1)
        #将（1，48，48，1）转换成为(1,1,48,48)
        face = np.expand_dims(face,0)
        face = np.expand_dims(face,0)

        # 人脸数据归一化，将像素值从0-255映射到0-1之间
        face = preprocess_input(face)
        new_face=torch.from_numpy(face)
        new_new_face = new_face.float().requires_grad_(False)
        
        # 调用我们训练好的表情识别模型，预测分类
        emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
        emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)

        try:
            # 获得出现次数最多的分类
            emotion_mode = mode(emotion_window)
        except:
            continue

        # 在矩形框上部，输出分类文字
        cv2.putText(frame,emotion_mode,(x,y-30), font, 1.0,(0,0,255),5,cv2.LINE_AA)
   st.image(frame,channels="BGR")

        


  
image_placeholder = st.empty()  # 创建空白块使得图片展示在同一位置    
f = st.file_uploader("上传本地视频进行人脸表情识别")  # 上传本地视频
if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    cap = cv2.VideoCapture(tfile.name)  # opencv打开文件
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.startWindowThread()
    #cv2.namedWindow('window_frame')

    if (cap.isOpened() == False):
        st.write("Error opening video stream or file")

    while (cap.isOpened()):
        success, frame = cap.read()
        #frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
        #frame = frame.copy()
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 获取当前帧中的全部人脸
        faces = face_detection.detectMultiScale(gray,1.3,5)
        # 对于所有发现的人脸
        for (x, y, w, h) in faces:
            # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
            cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

            # 获取人脸图像
            face = gray[y:y+h,x:x+w]

            try:
                # shape变为(48,48)
                face = cv2.resize(face,(48,48))
            except:
                continue

            # 扩充维度，shape变为(1,48,48,1)
            #将（1，48，48，1）转换成为(1,1,48,48)
            face = np.expand_dims(face,0)
            face = np.expand_dims(face,0)

            # 人脸数据归一化，将像素值从0-255映射到0-1之间
            face = preprocess_input(face)
            new_face=torch.from_numpy(face)
            new_new_face = new_face.float().requires_grad_(False)
            
            # 调用我们训练好的表情识别模型，预测分类
            emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
            emotion = emotion_labels[emotion_arg]

            emotion_window.append(emotion)

            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)

            try:
                # 获得出现次数最多的分类
                emotion_mode = mode(emotion_window)
            except:
                continue

            # 在矩形框上部，输出分类文字
            cv2.putText(frame,emotion_mode,(x,y-30), font, .7,(0,0,255),1,cv2.LINE_AA)
    
        if success:
            to_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(to_show, caption='Video')  # 将图片帧展示在同一位置得到视频效果
        else:
            break
    cap.release()




    
    


    
    
    
    
