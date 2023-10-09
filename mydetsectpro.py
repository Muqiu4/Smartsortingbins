# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import ctypes
import os
import platform
import sys
from pathlib import Path
import time
import torch
import serial
import cv2
from multiprocessing import Process, Queue, Value, shared_memory
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout, QTextEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QIcon
from PyQt5.QtCore import QTimer, QSize
import numpy as np
import multiprocessing
FILE = Path(__file__).resolve()#获取当前目录(detect.py)的(使用relsove)绝对路径,并将其赋值给变量FILE F:\yolov5-7.0\mydetect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取上一级目录 F:\yolov5-7.0
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径 F:\yolov5-7.0\mydetect.py
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
#定义了一共ui的类
#black = np.zeros((480, 640, 3), dtype=np.uint8)

class CameraUI(QWidget):
    def __init__(self):
        super().__init__()

        self.lock = multiprocessing.Lock()
        self.shared_int = multiprocessing.Value('i', 1)
        manager = multiprocessing.Manager()
        self.shared_dict = manager.dict()

        # 创建Manager对象用于创建共享变量
        self.shared_var = manager.Namespace()
        self.shared_var.np_frame = np.zeros((600, 800, 3), dtype=np.uint8)

        self.initUI()
        #self.detectUI()
        self.videoUI()
        self.show()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Unable to open camera", QMessageBox.Ok)
            self.close()
        ret2, frame2 = self.cap.read()
        if ret2:
            frame2 = cv2.resize(frame2, (800, 600))
            self.shared_var.np_frame = frame2

        self.video = cv2.VideoCapture('视频.mp4')
        if not self.video.isOpened():
            QMessageBox.critical(self, "错误", "无法加载视频", QMessageBox.Ok)
            self.close()

        self.detect_process = yolov5_detect(self.shared_dict, self.shared_var , self.lock,  self.shared_int)
        self.detect_process.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.open_video)
        self.timer.start()


    def initUI(self):
        self.setWindowTitle("智能分类垃圾桶")
        self.setGeometry(0, 0, 1080, 600)
        pixmap = QPixmap('窗口背景.png')
        # 创建调色板
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
    def detectUI(self):

        #满载区域
        self.title_label = QLabel(self)
        self.title_label.setText("智 能 分 类 垃 圾 桶")
        self.title_label.setGeometry(250, 550, 500, 50)
        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3; color: white;")

        self.labela = QLabel('四个垃圾桶满载情况', self)
        self.labela.setGeometry(800, 0, 210, 50)
        self.labela.setFont(QFont("Arial", 14))
        self.labela.setStyleSheet("color: yellow;")

        self.label0 = QLabel('0%', self)
        self.label0.setGeometry(900, 50, 100, 30)
        self.label0.setFont(QFont("Arial", 10))
        self.label0.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label1 = QLabel('0%', self)
        self.label1.setGeometry(900, 100, 100, 30)
        self.label1.setFont(QFont("Arial", 10))
        self.label1.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label2 = QLabel('0%', self)
        self.label2.setFont(QFont("Arial", 10))
        self.label2.setGeometry(900, 150, 100, 30)
        self.label2.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label3 = QLabel('0%', self)
        self.label3.setFont(QFont("Arial", 10))
        self.label3.setGeometry(900, 200, 100, 30)
        self.label3.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.btn1 = QLabel('其他垃圾:', self)
        self.btn1.setFont(QFont("Arial", 10))
        self.btn1.setStyleSheet("color: white;")
        self.btn1.setGeometry(800, 50, 100, 30)

        self.btn2 = QLabel('有害垃圾:', self)
        self.btn2.setFont(QFont("Arial", 10))
        self.btn2.setStyleSheet("color: white;")
        self.btn2.setGeometry(800, 100, 100, 30)

        self.btn3 = QLabel('厨余垃圾:', self)
        self.btn3.setFont(QFont("Arial", 10))
        self.btn3.setStyleSheet("color: white;")
        self.btn3.setGeometry(800, 150, 100, 30)

        self.btn4 = QLabel('可回收垃圾:', self)
        self.btn4.setFont(QFont("Arial", 10))
        self.btn4.setStyleSheet("color: white;")
        self.btn4.setGeometry(800, 200, 100, 30)
        #检测信息区域
        self.reportlabel = QLabel(self)
        self.reportlabel.setText("检测信息: ")
        self.reportlabel.setGeometry(0, 0, 150, 50)
        self.reportlabel.setFont(QFont("Arial", 16))
        self.reportlabel.setStyleSheet("color: white;")
        self.report = QLabel(self)
        self.report.setGeometry(0, 50, 210, 500)
        self.report.setFont(QFont("Arial", 13))
        self.report.setStyleSheet("color: black;"  # 设置文本显示区域样式
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 3;"
                                  "border-radius: 0;"
                                  "border-color: black;")

    def videoUI(self):
        # 显示区域
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1080, 600)
        self.label.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")


    def open_video(self):
        ret, frame = self.video.read()
        ret2, frame2 = self.cap.read()
        if ret2 and not self.shared_int.value:
            frame2 = cv2.resize(frame2, (800, 600))
            self.lock.acquire()
            self.shared_var.np_frame = frame2

            self.lock.release()
            self.shared_int.value = 1


        if not ret:
            # 到达视频结尾，将VideoCapture对象的位置设置为0，即重新播放视频
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.up_frame(frame,1080,600)
        if self.shared_dict:
            self.label.setGeometry(150, 0, 640, 480)
            self.timer.timeout.connect(self.open_detect)

    def open_detect(self):
        frame = self.shared_frame
        self.up_frame(frame,640,480)
    def up_frame(self,frame,vw,vh):
        frame = cv2.resize(frame,(1080,600))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(vw, vh, aspectRatioMode=1, transformMode=0)
        self.label.setPixmap(QPixmap.fromImage(p))



class yolov5_detect(multiprocessing.Process):
    def __init__(self, share_dict, share_frame,lock, shared_int ,weights='yolov5s.pt', device='cpu', conf_thres=0.8, iou_thres=1.0):
        super().__init__()
        self.shared_int = shared_int
        self.lock = lock
        self.shared_dict = share_dict
        self.shared_frame = share_frame
        self.retu = {}
        self.classes = ['else waste', 'hazardous', 'kitchen', 'recyclable']
        device = select_device(device=device)
        device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        self.device = device
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):
        while True:
            if self.shared_int.value:
                self.lock.acquire()
                self.frame =  self.shared_frame.np_frame
                self.lock.release()
                print('读取一张成功')
                self.shared_int.value = 0
                self.retu = {}
                count, C = 0, []
                lase_type = []
                t = time.time()
                self.frame = cv2.resize(self.frame, (800, 600))
                self.frame = self.frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 320):(800 // 2 + 320)]
                self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.img = torch.from_numpy(self.img).to(self.device).float() / 255.0
                self.img = self.img.permute(2, 0, 1).unsqueeze(0)
                if len(self.img.shape) == 3:
                    self.img = self.img[None]  # expand for batch dim如果张量的维度为 3，则在第 0 维上添加一个维度，以便将其扩展为批次大小为 1 的张量。
                self.pre = self.model.model(self.img, augment=False)  # 调用 YOLOv5 模型的 model 方法，对输入的图像或视频进行推理，并得到目标检测结果。
                self.pred = non_max_suppression(self.pre, self.conf_thres, self.iou_thres, None, False, max_det=10)
                if self.pred == None:
                    t = time.time() - t
                    return
                for det in self.pred[0]:
                    count += 1
                    if count > 1:
                        for [a, b] in C:
                            q = abs(int(det[0]) - a)
                            w = abs(int(det[1]) - b)
                            if q <= 5 and w <= 5:
                                ab = False
                                break
                            ab = True
                    else:
                        ab = True
                    if ab:
                        xyxy = (det[0], det[1], det[2], det[3])
                        cls = det[5]
                        conf = det[4]
                        ty = self.classes[int(cls)] + str(count) if int(det[5]) in lase_type else self.classes[int(cls)]
                        self.retu[ty] = list(int(x) for x in xyxy)
                        label = f'{self.classes[int(cls)]} {conf:.2f}'
                        cv2.rectangle(self.frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0),
                                      2)
                        cv2.putText(self.frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        lase_type.append(int(det[5]))
                    C.append([int(det[0]), int(det[1])])
                print(self.retu)
                t = time.time() - t

    def date_intaract(self):
        self.shared_dict = self.retu
        self.shared_frame = self.frame

if __name__=='__main__':

    app = QApplication(sys.argv)
    ex = CameraUI()
    sys.exit(app.exec_())

