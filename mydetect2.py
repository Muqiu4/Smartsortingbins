# YOLOv5 🚀 by Ultralytics, GPL-3.0 license




import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import cv2
from ultralytics.yolo.utils.ops import scale_coords

FILE = Path(__file__).resolve()#获取当前目录(detect.py)的(使用relsove)绝对路径,并将其赋值给变量FILE F:\yolov5-7.0\mydetect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取上一级目录 F:\yolov5-7.0
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径 F:\yolov5-7.0\mydetect.py
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

def yolov5_detect(weights='yolov5s.pt', device='cpu', conf_thres=0.8, iou_thres=1.0):
    t= time.time()

    cap = cv2.VideoCapture(1)
    classes = ['person', 'car', 'truck', 'bus']
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim如果张量的维度为 3，则在第 0 维上添加一个维度，以便将其扩展为批次大小为 1 的张量。
    pre = model.model(img, augment=False)  # 调用 YOLOv5 模型的 model 方法，对输入的图像或视频进行推理，并得到目标检测结果。
    pred = non_max_suppression(pre, conf_thres, iou_thres, None, False, max_det=10)

    for det in pred[0]:
        xyxy=(det[0],det[1],det[2],det[3])
        cls=det[5]
        conf=det[4]
        label = f'{classes[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame2', frame)
    t=time.time()-t
if __name__=='__main__':
    while True:
        yolov5_detect()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break