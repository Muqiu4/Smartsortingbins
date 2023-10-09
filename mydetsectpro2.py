# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import ctypes
import math
import os
import platform
import sys
from pathlib import Path
import time
import torch
import serial
import cv2
from PyQt5 import QtCore
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

class ReminderWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("警告")
        self.setGeometry(540, 300, 200, 30)
        layout = QVBoxLayout()
        self.label = QLabel("其他垃圾垃圾同已满载，请勿在投放|")
        self.label.setStyleSheet("color: red;")
        layout.addWidget(self.label)
        self.setLayout(layout)


    def set_data(self, data):
        rep = ''
        co = 1
        for a in data:
            if a == 'e':
                rep +='其他垃圾垃圾桶已满，请勿在投放！\n'
                co += 1
            elif a == 'h':
                rep +='有害垃圾垃圾桶已满，请勿在投放！\n'
                co += 1
            elif a == 'r':
                rep +='可回收垃圾垃圾桶已满，请勿在投放！\n'
                co += 1
            elif a == 'k':
                rep +='厨余垃圾垃圾桶已满，请勿在投放！\n'
                co += 1

        self.resize(200,15*co)
        self.label.setText(rep)
class CameraUI(QWidget):
    def __init__(self):
        super().__init__()
        self.detsign = False
        self.detcount = 0

        self.infor_sign = True
        self.lock_count = []
        self.tim_count = 1

        #串口
        self.queue = multiprocessing.Queue()
        self.shared_value = multiprocessing.Value('i', 0)
        self.shared_value2 = multiprocessing.Value('i', 0)
        manager = multiprocessing.Manager()
        self.shared_load = manager.list(['','', '', ''])
        # 创建Manager对象用于创建共享变量
        self.reminder_window = ReminderWindow()
        self.initUI()
        self.detectUI()
        self.videoUI()
        self.show()

        # 机器识别
        self.detect_process = yolov5_detect()
        if not self.detect_process.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头", QMessageBox.Ok)
            self.close()

        self.video = cv2.VideoCapture('垃圾分类宣传片.mp4')
        if not self.video.isOpened():
            QMessageBox.critical(self, "错误", "无法加载视频", QMessageBox.Ok)
            self.close()

        #串口
        ser_process = multiprocessing.Process(target=myserial, args=(self.queue, self.shared_value, self.shared_value2,self.shared_load,))
        ser_process.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.open_vidadet)
        self.timer.start(20)


    def initUI(self):
        self.setWindowTitle("智能分类垃圾桶")
        self.setGeometry(0, 0, 1080, 600)
        pixmap = QPixmap('窗口背景.png')
        # 创建调色板
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
    def detectUI(self):
        self.labeld = QLabel(self)
        self.labeld.setGeometry(220, 0, 640, 480)
        self.labeld.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")
        #满载区域
        self.title_label = QLabel(self)
        self.title_label.setText("智 能 分 类 垃 圾 桶")
        self.title_label.setGeometry(300, 500, 500, 50)
        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3; color: white;")

        self.labela = QLabel('四个垃圾桶满载情况', self)
        self.labela.setGeometry(870, 0, 210, 50)
        self.labela.setFont(QFont("Arial", 10))
        self.labela.setStyleSheet("color: yellow;")

        self.label0 = QLabel('未满', self)
        self.label0.setGeometry(970, 50, 100, 30)
        self.label0.setFont(QFont("Arial", 10))
        self.label0.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label1 = QLabel('未满', self)
        self.label1.setGeometry(970, 100, 100, 30)
        self.label1.setFont(QFont("Arial", 10))
        self.label1.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label2 = QLabel('未满', self)
        self.label2.setFont(QFont("Arial", 10))
        self.label2.setGeometry(970, 150, 100, 30)
        self.label2.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label3 = QLabel('未满', self)
        self.label3.setFont(QFont("Arial", 10))
        self.label3.setGeometry(970, 200, 100, 30)
        self.label3.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.btn1 = QLabel('其他垃圾:', self)
        self.btn1.setFont(QFont("Arial", 10))
        self.btn1.setStyleSheet("color: white;")
        self.btn1.setGeometry(870, 50, 100, 30)

        self.btn2 = QLabel('有害垃圾:', self)
        self.btn2.setFont(QFont("Arial", 10))
        self.btn2.setStyleSheet("color: white;")
        self.btn2.setGeometry(870, 100, 100, 30)

        self.btn3 = QLabel('厨余垃圾:', self)
        self.btn3.setFont(QFont("Arial", 10))
        self.btn3.setStyleSheet("color: white;")
        self.btn3.setGeometry(870, 150, 100, 30)

        self.btn4 = QLabel('可回收垃圾:', self)
        self.btn4.setFont(QFont("Arial", 10))
        self.btn4.setStyleSheet("color: white;")
        self.btn4.setGeometry(870, 200, 100, 30)
        #检测信息区域
        self.reportlabel = QLabel(self)
        self.reportlabel.setText("检测信息: ")
        self.reportlabel.setGeometry(0, 0, 150, 50)
        self.reportlabel.setFont(QFont("Arial", 16))
        self.reportlabel.setStyleSheet("color: white;font-weight: bold;")
        self.report = QLabel(self)
        self.report.setGeometry(0, 50, 220, 435)
        self.report.setFont(QFont("Arial", 13))
        self.report.setAlignment(QtCore.Qt.AlignTop)
        self.report.setStyleSheet("color: red;"  # 设置文本显示区域样式
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 5;"
                                  "border-radius: 0;"
                                  "border-color: black;")

    def videoUI(self):
        # 显示区域
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1080, 600)
        self.label.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")


    def open_vidadet(self):

        if self.tim_count%30==0 or self.detect_process.retu or self.detcount:
            self.detect_process.run()
            self.timer.setInterval(1)
            self.tim_count = 0

        else:
            self.timer.setInterval(20)

        self.tim_count += 1
        if self.detcount >0:
            self.detcount-=1
        if not self.detect_process.retu and not self.detcount:
            ret, frame = self.video.read()
            self.infor_sign = True
            self.detsign = False
            self.label.show()
        else:
            ret ,self.detsign = True ,True
            if self.detect_process.retu:
                self.detcount = 100
            frame = self.detect_process.frame
            self.label.hide()

        if not ret:
            # 到达视频结尾，将VideoCapture对象的位置设置为0，即重新播放视频
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.up_frame(frame,1080,600) if not self.detsign else self.up_frame(frame, 640, 480)



    def up_frame(self,frame,vw,vh):

        frame = cv2.resize(frame,(vw,vh))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(vw, vh, aspectRatioMode=1, transformMode=0)
        self.label.setPixmap(QPixmap.fromImage(p)) if not self.detsign else self.labeld.setPixmap(QPixmap.fromImage(p))
        if self.detsign:
            self.detectinfor() if self.infor_sign else "pass"
            self.dp_infor()
        else:
            self.load_change()
    def detectinfor(self):
        count = 1
        if "recyclable" in self.detect_process.retu:
            self.detect_process.retu = {k: self.detect_process.retu[k] for k in sorted(self.detect_process.retu, key=custom_sort_func)}
        # 识别到的垃圾种类，序号，状态，中心坐标
        self.waste_type = [''] * 10
        self.waste_num = []
        self.waste_statu = [''] * 10
        self.waste_address = np.empty(10, dtype=object)

        for mkey,mvalue in self.detect_process.retu.items():

            if mkey[0]=='e':
                type = '其他垃圾'
            elif mkey[0]=='h':
                type = '有害垃圾'
            elif mkey[0] == 'k':
                type = '厨余垃圾'
            elif mkey[0] == 'r':
                type = '可回收垃圾'
            else:
                pass
            self.waste_num.append(count)
            self.waste_type[count] = type
            self.waste_statu[count] = '待处理'
            self.waste_address[count] = [(mvalue[0]+mvalue[2])//2,(mvalue[1]+mvalue[3])//2]
            count +=1
        self.lock_count.append(count) if self.detect_process.retu else self.lock_count.clear()

        if len(self.lock_count)>20:
            self.infor_sign = False if max(self.lock_count)==min(self.lock_count) else True
            self.lock_count.clear()
    def dp_infor(self):
        if self.infor_sign:
            self.st=1
            self.reportext = ['']*10
            for i in self.waste_num:
                self.reportext[i-1]=f'{i}--'+self.waste_type[i] +'--'+self.waste_statu[i]
                self.num = i
        else:
            if self.st:
                self.queue.put(self.detect_process.retu)
                self.st=0
        self.change()
        self.report.setText(self.reportext[0]+'\n'+self.reportext[1]+'\n'+self.reportext[2]+'\n'+self.reportext[3]+'\n')
    def load_change(self):


        self.reminder_window.set_data(self.shared_load)
        if self.shared_load[0] == 'e' or self.shared_load[1] == 'h' or self.shared_load[3] == 'k' or self.shared_load[2] == 'r':
            self.reminder_window.show()
        else:
            self.reminder_window.close()


    def change(self):
        if self.shared_load[0] == 'e':
            self.label0.setText("以满载")
            self.label0.setStyleSheet("color: red;"
                                      "background-color: white;")

        else:
            self.label0.setText("未满")
            self.label0.setStyleSheet("color: green;"
                                      "background-color: white;")
        if self.shared_load[1] == 'h':
            self.label1.setText("已满载")
            self.label1.setStyleSheet("color: red;"
                                      "background-color: white;")

        else:
            self.label1.setText("未满")
            self.label1.setStyleSheet("color: green;"
                                      "background-color: white;")
        if self.shared_load[3] == 'k':
            self.label2.setText("以满载")
            self.label2.setStyleSheet("color: red;"
                                      "background-color: white;")
        else:
            self.label2.setText("未满")
            self.label2.setStyleSheet("color: green;"
                                      "background-color: white;")
        if self.shared_load[2] == 'r':
            self.label3.setText("以满载")
            self.label3.setStyleSheet("color: red;"
                                      "background-color: white;")
        else:
            self.label3.setText("未满")
            self.label3.setStyleSheet("color: green;"
                                      "background-color: white;")
        if self.shared_value.value:
            new_str = self.reportext[self.shared_value.value - 1].replace('待处理', '处理中')
            self.reportext[self.shared_value.value - 1] = new_str
        if self.shared_value2.value:
            new_str = self.reportext[self.shared_value2.value - 1].replace('处理中', '-ok')
            self.reportext[self.shared_value2.value - 1] = new_str
            if int(self.shared_value2.value) == self.num:
                self.shared_value.value,self.shared_value2.value= 0,0
                print('垃圾处理完毕')
                self.infor_sign = True



def myserial(queue,shared_value,shared_value2,shared_load):
    ser = serial.Serial('COM5', 9600, timeout=1)# 打开串口
    time.sleep(5)
    #串口打开成功
    print('串口打开成功')

    ser_status = 0
    ace = []#平均值
    count ,coord= 0,[]
    try:
        while True:
            if not queue.empty():
                data = queue.get()
                print('data',data)
                waste_type,waste_coor = list(data.keys()),list(data.values())

                for (x0, y0, x1, y1) in  waste_coor:
                    ac = str((x0 + x1 + y0 + y1)//4)
                    x0 = str(x0)
                    y0 = str(y0)
                    x1 = str(x1)
                    y1 = str(y1)

                    if len(x0) < 3:
                        x0 = '0' * (3 - len(x0)) + x0
                    if len(x1) < 3:
                        x1 = '0' * (3 - len(x1)) + x1
                    if len(y0) < 3:
                        y0 = '0' * (3 - len(y0)) + y0
                    if len(y1) < 3:
                        y1 = '0' * (3 - len(y1)) + y1
                    if len(ac) < 3:
                        ac = '0' * (3 - len(ac)) + ac
                    co = x0+y0+x1+y1
                    ace.append(ac)
                    print('ac:',ac)
                    coord.append(co)
                print("coord:",coord)
                for t,c,a in zip(waste_type,coord,ace):
                   # print('tca:',t,c,a)
                    while True:
                        if len(waste_type) - count == 1:
                            count = 0
                        ser.write(str(len(waste_type)).encode()) if len(waste_type) == 1 else ser.write(str(len(waste_type) - count).encode())

                        time.sleep(0.1)
                        ser.write(t[0].encode())
                        time.sleep(0.1)
                        ser.write(c.encode())
                        time.sleep(0.1)
                        ser.write(a.encode())
                        time.sleep(0.1)
                        print('ac',a)
                        print('发送一次成功')
                        ser.write(b'o')
                        while True:
                            T = time.time()
                            res = ser.read()

                            if res:
                                res = res.decode()
                                #print("res", res)
                                if res == '1':
                                    print('成功')
                                    shared_value.value += 1
                                    count +=1
                                elif res == '2':
                                    ser_status = 1
                                    shared_value2.value = shared_value.value
                                    break
                                elif res == '0':
                                    print('发送错误,重发')
                                    break
                                elif res == 'e':
                                    shared_load[0] = 'e'
                                elif res == 'h':
                                    shared_load[1] = 'h'
                                elif res == 'r':
                                    shared_load[2] = 'r'
                                elif res == 'k':
                                    shared_load[3] = 'k'
                                elif res == 'c':
                                    shared_load[0] = ''
                                elif res == 'b':
                                    shared_load[1] = ''
                                elif res == 'a':
                                    shared_load[2] = ''
                                elif res == 'd':
                                    shared_load[3] = ''


                            T=time.time()-T
                            if T>2:
                                break
                        print('发送下一个垃圾')
                        if ser_status:
                            ser_status = 0
                            break
                ace.clear()
                data.clear()
                coord.clear()
            else:
                res_=ser.read()
                res_ = res_.decode()

                if res_ == 'e':
                    shared_load[0] = 'e'
                elif res_ == 'h':
                    shared_load[1] = 'h'
                elif res_ == 'r':
                    shared_load[2] = 'r'
                elif res_ == 'k':
                    shared_load[3] = 'k'
                elif res_ == 'c':
                    shared_load[0] = ''
                elif res_ == 'b':
                    shared_load[1] = ''
                elif res_ == 'a':
                    shared_load[2] = ''
                elif res_ == 'd':
                    shared_load[3] = ''
    finally:
        # 关闭串口
        if ser.is_open:
            ser.close()

class yolov5_detect():
    def __init__(self ,weights='best.pt', device='0', conf_thres=0.8, iou_thres=1.0):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.retu = {}
        self.classes = ['else waste', 'hazardous', 'kitchen', 'recyclable']
        device = select_device(device=device)
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        self.device = device
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):

        self.retu = {}
        count, C = 0, []
        lase_type = []
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
        self.pred = non_max_suppression(self.pre, self.conf_thres, self.iou_thres, None, False, max_det=20)
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
                cv2.rectangle(self.frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(self.frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                lase_type.append(int(det[5]))
            C.append([int(det[0]), int(det[1])])
        print('retu:',self.retu)
        t = time.time() - t

def custom_sort_func(key):
    if key.startswith('recyclable'):
        return (0, key)  # 返回一个元组，第一个元素为0，确保f开头的键排在前面
    else:
        return (1, key)  # 非f开头的键排在后面

if __name__=='__main__':

    app = QApplication(sys.argv)
    ex = CameraUI()
    sys.exit(app.exec_())

