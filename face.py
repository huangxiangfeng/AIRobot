#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
人脸检测DEMO
anaconda 
python<=3.5
conda install -c menpo opencv3
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding = 'utf8')

import cv2
import numpy as np
import align.detect_face
import tensorflow as tf
import time
import recognise.facenet as facenet
import matplotlib.pyplot as plt
import os
import mtcnn
from scipy import misc

#人脸识别
from mtcnn.MtcnnDetector import MtcnnDetector
from mtcnn.detector import Detector
from mtcnn.fcn_detector import FcnDetector
from mtcnn.mtcnn_model import P_Net, R_Net, O_Net
from mtcnn.loader import TestLoader

#对话机器人
import re
from chatbot import chatbot

#截取
class MTCNNAline():
    def __init__(self):

        test_mode = "ONet"
        thresh = [0.9, 0.6, 0.7]
        min_face_size = 24
        stride = 2
        slide_window = False
        shuffle = False
        detectors = [None, None, None]
        prefix = ['model/MTCNN_model/PNet_landmark/PNet', 'model/MTCNN_model/RNet_landmark/RNet', 'model/MTCNN_model/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        batch_size = [2048, 256, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        # load pnet model
        if slide_window:
            PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
        else:
            PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet

        # load rnet model
        if test_mode in ["RNet", "ONet"]:
            RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
            detectors[1] = RNet

        # load onet model
        if test_mode == "ONet":
            ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
            detectors[2] = ONet

        self.mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                    stride=stride, threshold=thresh, slide_window=slide_window)
    def detect_face(self, img):
        bounding_boxes, landmark = self.mtcnn_detector.detect(img)
        return bounding_boxes
    def detect_img(self, img, margin=44):
        bounding_boxes = self.detect_face(img)
        if len(bounding_boxes)==0:
            return [], []
        img_list = [None] * len(bounding_boxes)
        for i, bounding_box in enumerate(bounding_boxes):
            det = np.squeeze(bounding_box[0:4])
            bb = np.zeros(4, dtype=np.int32)
            img_size = np.asarray(img.shape)[0:2]
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = np.squeeze(prewhitened)
        images = np.stack(img_list)
        return bounding_boxes, images


class FaceAline():
    def __init__(self):
        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor
        
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=100)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)
    def detect_face(self, img):
        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return bounding_boxes
    def detect_img(self, img, margin=44):
        bounding_boxes = self.detect_face(img)
        if len(bounding_boxes)==0:
            return [], []
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        img_size = np.asarray(img.shape)[0:2]
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        return bounding_boxes, [prewhitened]
        
#识别
class FaceDetection():
    def __init__(self):
        self.aline_class = MTCNNAline()
        self.aline = self.aline_class.detect_img
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=100)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                # Load the model
                facenet.load_model("model/facenet")
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.reco = lambda img: sess.run(embeddings, feed_dict={images_placeholder: img, phase_train_placeholder:False })
    def recognise(self, img):
        bbox, img_p = self.aline_class.detect_img(img)
        if len(bbox)==0:
            return [], []
        vects = self.reco(img_p)
        names = []
        for itr_v in vects:
            name = "NotKnown"
            for itr in self.vects:
                for itrb in itr[1]:
                    dist = np.sqrt(np.sum(np.square(np.subtract(itr_v, itrb))))
                    if(dist < 1):
                        name = itr[0]
            names.append(name)
        return bbox, names
    def get_vect(self, dirs, name="obama"):
        files = os.listdir(dirs)
        pth = []
        for itr in files:
            pth.append(dirs+'/'+itr)
        images = self.load_detect(pth)
        vect = self.reco(images)
        np.savez("data/vect/" + name + ".npz", name = name, vect = np.array(vect))
    def load_vect(self, dirs):
        files = os.listdir(dirs)
        pth = []
        for itr in files:
            pth.append(dirs+'/'+itr)
        self.vects = []
        for itr in pth:
            tdata = np.load(itr)
            self.vects.append([tdata["name"], tdata["vect"]])
    def load_detect(self, image_paths, margin=44, image_size=160):
        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=100)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        #print(image_paths)
        nrof_samples = len(image_paths)
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
        images = np.stack(img_list)
        return images
class Faces():
    def __init__(self):
        pass
    def load_aline(self):
        self.detect_face = FaceAline().detect_face
    def load_detect(self):
        self.recognise = FaceDetection()
        self.recognise.load_vect("data/vect")
    def get_cam(self):
        cap = cv2.VideoCapture(0)
        getValidityFace = False
        userName = "NotKnown"
        while(not getValidityFace):
            # get a frame
            ret, frame = cap.read()
            cv2.imshow("capture", frame)
            #boxes = self.detect_face(frame)
            boxes, names = self.recognise.recognise(frame)
            time.sleep(0.1)
            if len(names)>0:
                userName = names[0]
            if userName != "NotKnown":
                getValidityFace = True
                cv2.rectangle(frame, (int(boxes[0][0]),int(boxes[0][1])),(int(boxes[0][2]),int(boxes[0][3])),(100,100,100),7)
                cv2.imshow("capture", frame)
                time.sleep(0.2)

                #cv2.putText(frame,str(name),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(100, 100, 100))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        print("exit")
        cap.release()
        cv2.destroyAllWindows()
        return userName
    def get_video(self):
        videoCapture = cv2.VideoCapture('data/obama.mp4')
        #读帧  
        success, frame = videoCapture.read() 
        cont = 0
        while success :
            if cont%10==0: 
                boxes, names = self.recognise.recognise(frame)
            cont += 1
            #time.sleep(0.1)
            for bbox, name in zip(boxes, names):
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(100,100,100),7)
                cv2.putText(frame,str(name),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(100, 100, 100))
            cv2.imshow("Oto Video", frame) #显示  
            cv2.waitKey(40) #延迟  
            success, frame = videoCapture.read() #获取下一帧  
    def one_img(self, imgdir):
        frame = cv2.imread(imgdir)
        boxes, names = self.recognise.recognise(frame)
        #time.sleep(0.1)
        for bbox, name in zip(boxes, names):
            cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(100,100,100),7)
            cv2.putText(frame,str(name),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(100, 100, 100))
        cv2.imshow("ctd",frame)
        cv2.waitKey(0)

    def getNameFromeImg(self, imgdir):
        frame = cv2.imread(imgdir)
        boxes, names = self.recognise.recognise(frame)
        return names[0]

import tkinter as tk
from tkinter import *

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.username="me"

    def createWidgets(self):
        # 人脸识别
        self.faceButton = tk.Button(self, text='人脸识别', command=self.faceRecoginise)
        self.faceButton.pack()

        # 语音识别-语音聊天
        self.voiceButton = tk.Checkbutton(self, text='语音聊天')
        self.voiceButton.pack()

        # 文本聊天对话内容
        self.dialogueContent = tk.Text(self, width=80,height=20)
        self.dialogueContent.pack()
        self.dialogueContent.insert(INSERT,"bot: ",'big')
        self.dialogueContent.insert(INSERT,"first !\n")

        # 发言文本框
        self.dialogueInput = tk.Entry(self)
        self.dialogueInput.pack()
        # 发言文本发送按钮
        self.dialogueSendButton = tk.Button(self, text='发送', command=self.dialogueSend)
        self.dialogueSendButton.pack()

    def dialogueSend(self):
        msgInput = self.dialogueInput.get()
        self.dialogueInput.delete(0, END)
        print(self.username)
        self.dialogueContent.insert(INSERT,'%s%s%s%s' % (self.username,':',msgInput,'\n'))
        self.dialogueContent.insert(INSERT,'%s%s%s' % ('bot:',bot.daemonPredict(msgInput),'\n'))

    def faceRecoginise(self):
        self.username = fr.get_cam()
        if self.username == '':
            messagebox.showinfo('登陆成功', '未识别头像')
        else:
            messagebox.showinfo('登陆成功', '欢迎回来, 亲爱的%s' % self.username)
            self.master.title('聊天机器人正在和 %s 聊天' % self.username)


if __name__=="__main__":
    root = tk.Tk()
    root.geometry('500x600')#窗体大小
    root.resizable(False, False)#固定窗体
    app = Application(master=root)
    # 设置窗口标题:
    app.master.title('聊天机器人')

    fr = Faces()
    fr.load_aline()
    fr.load_detect()

    bot = chatbot.Chatbot()
    #bot.main(['--test', 'daemon', '--modelTag', 'xiaohuang'])
    bot.main(['--test', 'daemon'])

    # 主消息循环:
    app.mainloop()