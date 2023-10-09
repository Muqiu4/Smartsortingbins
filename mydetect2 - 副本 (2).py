# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import serial
import cv2
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout, QTextEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QIcon
from PyQt5.QtCore import QTimer, QSize
import numpy as np
from multiprocessing import Process
FILE = Path(__file__).resolve()#获取当前目录(detect.py)的(使用relsove)绝对路径,并将其赋值给变量FILE F:\yolov5-7.0\mydetect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取上一级目录 F:\yolov5-7.0
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径 F:\yolov5-7.0\mydetect.py
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
#定义了一共ui的类
black = np.zeros((480, 640, 3), dtype=np.uint8)

"""def reason(s,):
    a = yolov5_detect()
    a.start_detect()
    return a.retu"""


class CameraUI(QWidget):
    def __init__(self,queue):
        super().__init__()
        self.queue=queue
        self.detecting=yolov5_detect()
        #self.ser = serial.Serial('/dev/ttyUSB0', 9600)  # 根据实际情况修改串口名和波特率
        self.check_box = None
        self.initUI()
        self.camera =  self.detecting.cap
        self.video = cv2.VideoCapture('视频.mp4')
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "Unable to open camera", QMessageBox.Ok)
            self.close()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.show_video=False
        self.show_camera = False
        self.show_detect = False
        self.video_detect = False
        self.label.setPixmap(QPixmap.fromImage(QImage(black.data, 640, 480, QImage.Format_RGB888)))

    def initUI(self):
        self.setWindowTitle("Camera UI")
        self.setcheckbox()
        self.setGeometry(0, 0, 1080, 600)
        pixmap = QPixmap('窗口背景.png')
        but0_pixmap = QPixmap('button0.png')
        but1_pixmap = QPixmap('button1.png')
        but2_pixmap = QPixmap('button3.png')
        close_pixmap = QPixmap('关闭按钮.png')
        # 创建调色板
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        icon = QIcon(but0_pixmap)
        icon1 = QIcon(but1_pixmap)
        icon2 = QIcon(but2_pixmap)
        icon3 = QIcon(close_pixmap)
        # 设置窗口的调色板
        self.setPalette(palette)


        self.label = QLabel(self)
        self.label.setGeometry(150, 0, 640, 480)
        self.label.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")

        self.fpslabel = QLabel(self)
        self.fpslabel.setText("FPS:  ")
        self.fpslabel.setGeometry(1030, 570, 50, 30)
        self.fpslabel.setFont(QFont("Arial", 8))
        self.fpslabel.setStyleSheet("color: white;")

        self.reportlabel = QLabel(self)
        self.reportlabel.setText("检测信息: ")
        self.reportlabel.setGeometry(10, 480, 150, 50)
        self.reportlabel.setFont(QFont("Arial", 16))
        self.reportlabel.setStyleSheet("color: white;")
        self.report = QLabel(self)
        self.report.setGeometry(150, 480, 800, 50)
        self.report.setFont(QFont("Arial", 13))
        self.report.setStyleSheet("color: black;"# 设置文本显示区域样式
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 2;"
                                  "border-radius: 6;"
                                  "border-color: white;")
        self.waste_infor()

        self.button = QPushButton('Open Cam',self)
        self.button1 = QPushButton("", self)
        self.button2 = QPushButton("open video", self)
        self.button3 = QPushButton("sta detect", self)
        self.button.setGeometry(0, 0, 150, 50)
        self.button2.setGeometry(0, 200, 150, 50)
        self.button1.setGeometry(0, 300, 50, 50)
        self.button3.setGeometry(0, 100, 150, 50)

        self.setButtonStyle(self.button,"white","black","yellow")
        self.button.setFont(QFont("Arial", 12))
        self.button.setIcon(icon)
        self.button.setIconSize(QSize(30,30))


        self.setButtonStyle(self.button2, "white", "black", "yellow")
        self.button2.setFont(QFont("Arial", 12))
        self.button2.setIcon(icon1)
        self.button2.setIconSize(QSize(30, 30))

        self.setButtonStyle(self.button3, "white", "black", "yellow")
        self.button3.setFont(QFont("Arial", 12))
        self.button3.setIcon(icon2)
        self.button3.setIconSize(QSize(30, 30))

        self.button1.setStyleSheet("border-radius: 3;")
        self.button1.setIcon(icon3)
        self.button1.setIconSize(QSize(50, 50))

        self.button.clicked.connect(self.toggle_camera)
        self.button1.clicked.connect(self.close)
        self.button2.clicked.connect(self.toggle_video)
        self.button3.clicked.connect(self.toggle_detect)
        self.loading()
        self.show()
    def loading(self):
        self.title_label = QLabel(self)
        self.title_label.setText("智 能 分 类 垃 圾 桶")
        self.title_label.setGeometry(0, 550, 500, 50)
        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3; color: white;")

        self.labela = QLabel('四个垃圾桶满载情况', self)
        self.labela.setGeometry(800, 0, 210, 50)
        self.labela.setFont(QFont("Arial", 14))
        self.labela.setStyleSheet("color: yellow;")

        self.label0 = QLabel('0%',self)
        self.label0.setGeometry(900, 50, 100, 30)
        self.label0.setFont(QFont("Arial", 10))
        self.label0.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label1 = QLabel('0%',self)
        self.label1.setGeometry(900, 100, 100, 30)
        self.label1.setFont(QFont("Arial", 10))
        self.label1.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label2 = QLabel('0%',self)
        self.label2.setFont(QFont("Arial", 10))
        self.label2.setGeometry(900, 150, 100, 30)
        self.label2.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label3 = QLabel('0%',self)
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

    def but_statue(self):
        self.button.setText('Open Cam') if not self.show_camera else self.button.setText('Close Cam')
        self.setButtonStyle(self.button, "white", "black", "yellow") if not self.show_camera else  self.setButtonStyle(self.button, "gray", "black", "yellow")
        self.button2.setText('Open Video') if not self.show_camera else self.button.setText('Close Video')
        self.setButtonStyle(self.button2, "white", "black", "yellow") if not self.show_video else self.setButtonStyle(self.button2, "gray", "black", "yellow")
        self.button3.setText('sta detect') if not self.show_camera else self.button.setText('Clo detect')
        self.setButtonStyle(self.button3, "white", "black", "yellow") if not self.show_detect else self.setButtonStyle(self.button3, "gray", "black", "yellow")
    def toggle_detect(self):
        if self.show_detect:
            self.show_camera = True
            self.show_detect = False
            self.show_video = False
            self.but_statue()
            self.waste_infor()
        else:
            self.show_detect = True
            self.show_camera = False
            self.show_video = False
            self.but_statue()
            self.timer.start()
    def toggle_video(self):
        if self.show_video:
            self.show_video = False
            self.but_statue()
            self.timer.stop()
            self.fpslabel.setText("FPS:  ")
            self.label.setPixmap(QPixmap.fromImage(QImage(black.data, 640, 480, QImage.Format_RGB888)))
        else:
            self.show_video = True
            self.show_camera = False
            self.show_detect = False
            self.but_statue()
            self.timer.start(20)
    def toggle_camera(self):
        if self.show_camera:
            self.show_camera = False
            self.show_detect = False
            self.but_statue()
            self.timer.stop()
            self.fpslabel.setText("FPS:  ")
            self.label.setPixmap(QPixmap.fromImage(QImage(black.data, 640, 480, QImage.Format_RGB888)))
        else:
            self.show_camera = True
            self.show_video = False
            self.show_detect = False
            self.but_statue()
            self.timer.start()

    def update_frame(self):
        if not self.video_detect:
            if self.show_camera:
                ret, frame = self.camera.read()
            if self.show_video:
                ret, frame = self.video.read()
                if not ret:
                    # 到达视频结尾，将VideoCapture对象的位置设置为0，即重新播放视频
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
            if self.show_detect:
                self.start_detect()
                self.waste_infor()
                ret=False
            if ret:
                frame = cv2.resize(frame, (800, 600))
                frame = frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 320):(800 // 2 + 320)]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, aspectRatioMode=1, transformMode=0)
                self.label.setPixmap(QPixmap.fromImage(p))
                fps = self.camera.get(cv2.CAP_PROP_FPS)
                self.fpslabel.setText("FPS:" + str(int(fps)))
        else:
            self.vidanddec()
            self.waste_infor()
    def start_detect(self):
        if self.show_detect:
            self.detecting.start_detect()
            frame=self.detecting.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, aspectRatioMode=1, transformMode=0)
            self.label.setPixmap(QPixmap.fromImage(p))


    def setcheckbox(self):
        self.check_box = QCheckBox('主机', self)
        self.check_box.setGeometry(1020, 0, 50, 35)  # 设置复选框的位置和大小
        self.check_box.setStyleSheet("color: white;") # 设置字体颜色
        self.check_box.setChecked(False)  # 设置复选框默认为未选中状态
        self.check_box.stateChanged.connect(self.update_label)

    def update_label(self, state):
        if state==2:#隐藏按钮
            self.button.hide()
            self.button1.hide()
            self.button2.hide()
            self.button3.hide()
            self.show_video = True
            self.show_camera = False
            self.show_detect = False
            self.video_detect = True
            self.timer.start()

        else:
            self.video_detect = False

            self.button.show()
            self.button1.show()
            self.button2.show()
            self.button3.show()

    def waste_infor(self):
        if self.detecting.retu:
            oth_count = sum(1 for key in self.detecting.retu.keys() if 'else waste' in key)
            haz_count = sum(1 for key in self.detecting.retu.keys() if 'hazardous' in key)
            kit_count = sum(1 for key in self.detecting.retu.keys() if 'kitchen' in key)
            rec_count = sum(1 for key in self.detecting.retu.keys() if 'recyclable' in key)
            information = f'已检测到{len(list(self.detecting.retu.keys()))}个垃圾: ' + f'{oth_count}个其他垃圾，'+f'{haz_count}个有害垃圾，'+f'{rec_count}个可回收垃圾，'+ f'{kit_count}个厨余垃圾，'
            retu =  {key[0]: value for key, value in self.detecting.retu.items()}
            self.queue.put(retu)
            self.detecting.retu={}
        else:
            information = '未检测到垃圾'
        self.report.setText(information)
    def vidanddec(self):
        ret=None
        self.detecting.start_detect()
        if not self.detecting.retu:
            if self.show_video:
                ret, frame = self.video.read()
                if not ret:
                    # 到达视频结尾，将VideoCapture对象的位置设置为0，即重新播放视频
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
        else:
            ret = False
            frame = self.detecting.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, aspectRatioMode=1, transformMode=0)
            self.label.setPixmap(QPixmap.fromImage(p))
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame = frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 320):(800 // 2 + 320)]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, aspectRatioMode=1, transformMode=0)
            self.label.setPixmap(QPixmap.fromImage(p))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            self.fpslabel.setText("FPS:" + str(int(fps)))


    def setButtonStyle(self, button, background_color, font_color, border_color,border_width=4,\
                       radius=25, shape="round"):
        """
        设置按钮的样式
        :param button: QPushButton对象
        :param background_color: 背景颜色，默认为白色
        :param font_color: 字体颜色，默认为黑色
        :param border_width: 边框宽度，默认为2像素
        :param border_color: 边框颜色，默认为米色
        :param radius: 圆角半径，默认为10像素
        :param shape: 按钮形状，默认为圆形
        """
        if shape == "round":
            button.setStyleSheet("color: %s;"
                                 "background-color: %s;"
                                 "border-style: outset;"
                                 "border-width: %dpx;"
                                 "border-radius: %dpx;"
                                 "border-color: %s;" % (font_color, background_color, border_width, radius, border_color))
#一下为识别的代码
def myserial(queue):
    ser = serial.Serial('COM1', 9600, timeout=1)# 打开串口
    #串口打开成功
    statue = True
    count ,coord= 0,[]
    try:
        while True:
            for i in range(3):
                data = queue.get()
                waste_type,waste_coor = list(data.key()),list(data.value())
                for type,(x0,y0,x1,y1) in waste_type,waste_coor:
                    x ,y = str((x0 + x1) // 2),str((y0 + y1) // 2)
                    x = '00' + str(x) if len(str(x)) == 1 else x
                    y = '00' + str(y) if len(str(y)) == 1 else y
                    x = '0' + str(x) if len(str(x)) == 2 else x
                    y = '0' + str(y) if len(str(y)) == 2 else y
                    co = x+y
                    coord.append(co)

            for t,c in waste_type,coord:
                ser.write(t.encode())
                time.sleep(0.1)
                ser.write(c.encode())
                time.sleep(0.1)
            ser.write(b'o')
            while True:
                res = ser.read()
                res = res.decode()
                if res == '1':
                    print('发送一次成功')
                    break
                elif res == '2':
                    print('发送失败，准备重发')
                    break              # 在此处执行其他操作
    finally:
        # 关闭串口
        if ser.is_open:
            ser.close()
class yolov5_detect():
    def __init__(self,weights='yolov5s.pt', device='cpu', conf_thres=0.8, iou_thres=1.0):
        self.cap = cv2.VideoCapture(0)
        self.retu = {}
        self.classes = ['else waste', 'hazardous', 'kitchen', 'recyclable']
        device = select_device(device=device)
        #device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        self.device=device
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    def start_detect(self):
        self.retu = {}
        count,C=0,[]
        lase_type=[]
        t = time.time()
        self.ret, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, (800, 600))
        self.frame = self.frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 320):(800 // 2 + 320)]
        self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.img = torch.from_numpy(self.img).to(self.device).float() / 255.0
        self.img = self.img.permute(2, 0, 1).unsqueeze(0)
        if len(self.img.shape) == 3:
            self.img = self.img[None]  # expand for batch dim如果张量的维度为 3，则在第 0 维上添加一个维度，以便将其扩展为批次大小为 1 的张量。
        self.pre = self.model.model(self.img, augment=False)  # 调用 YOLOv5 模型的 model 方法，对输入的图像或视频进行推理，并得到目标检测结果。
        self.pred = non_max_suppression(self.pre, self.conf_thres, self.iou_thres, None, False, max_det=10)
        if self.pred==None:
            t = time.time() - t
            return
        for det in self.pred[0]:
            count += 1
            if count>1:
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
                ty =self.classes[int(cls)]+str(count)if int(det[5]) in lase_type else self.classes[int(cls)]
                self.retu[ty]=list(int(x) for x in xyxy)
                label = f'{self.classes[int(cls)]} {conf:.2f}'
                cv2.rectangle(self.frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(self.frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                lase_type.append(int(det[5]))
            C.append([int(det[0]), int(det[1])])
        print(self.retu)
        t = time.time() - t


if __name__=='__main__':
    queue = Queue()
    p = Process(target=myserial, args=(queue,))

    app = QApplication(sys.argv)
    ex = CameraUI(queue)
    sys.exit(app.exec_())