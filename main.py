import sys
import cv2
from IPython.external.qt_for_kernel import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, \
    QSizePolicy, QFrame, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QLinearGradient, QPalette, QBrush
from PyQt5.QtCore import Qt, QTimer
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

weights = "yolov5s.pt"
device = select_device(device='cpu')
model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
conf_thres = 0.9
iou_thres = 1.0
min_detection_frames = 10  # 最小连续检测帧数


class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super(VideoLabel, self).__init__(parent)

    def paintEvent(self, event):
        # 在视频框上绘制物体检测结果
        super(VideoLabel, self).paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red)
        painter.setPen(pen)

        objects = getattr(self, 'objects', [])
        for obj in objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            painter.drawText(bbox[0], bbox[1] - 25, f"{class_name} ({center_x}, {center_y})")
            painter.drawRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            painter.drawEllipse(center_x, center_y, 3, 3)


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("垃圾分类")
        self.setGeometry(400, 50, 1080, 600)

        background_image = QPixmap("pt/背景图片.png")
        self.background_label = QLabel(self)
        self.background_label.setPixmap(background_image)
        self.background_label.setGeometry(0, 0, background_image.width(), background_image.height())

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QHBoxLayout(self.central_widget)

        # 添加标题
        title_layout = QVBoxLayout()
        self.title_label = QLabel(self.central_widget)
        self.title_label.setText("智\n能\n分\n类\n垃\n圾\n桶")

        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3;")

        # 自定义绘制函数
        def paintEvent(event):
            painter = QPainter(self.title_label)
            gradient = QLinearGradient(0, 0, self.title_label.width(), self.title_label.height())
            gradient.setColorAt(0, QColor("#ffffff"))  # 起始颜色设置为白色
            gradient.setColorAt(1, QColor("#d3d3d3"))  # 结束颜色设置为浅灰色或银色
            painter.setPen(QPen(gradient, 0))
            painter.drawText(event.rect(), Qt.AlignCenter, self.title_label.text())

        # 重写paintEvent方法
        self.title_label.paintEvent = paintEvent

        title_layout.addWidget(self.title_label)
        layout.addLayout(title_layout)

        # 视频框布局
        video_layout = QVBoxLayout()

        # 左侧视频框
        self.label_camera = VideoLabel(self.central_widget)
        self.label_camera.setStyleSheet("background-color: transparent; border-radius: 5px; border: 2px solid #d3d3d3;")
        self.label_camera.setFixedSize(640, 480)  # 设置固定大小
        video_layout.addWidget(self.label_camera)

        # 右侧视频框
        self.label_mp4 = VideoLabel(self.central_widget)
        self.label_mp4.setStyleSheet("background-color: transparent; border-radius: 5px; border: 2px solid #d3d3d3;")
        self.label_mp4.setFixedSize(640, 480)  # 设置固定大小
        video_layout.addWidget(self.label_mp4)

        layout.addLayout(video_layout)

        # 右侧垂直布局
        right_layout = QVBoxLayout()

        # 检测信息框
        self.result_text = QTextEdit(self.central_widget)
        self.result_text.setFixedSize(600, 485)  # 设置宽度为600，高度为500
        self.result_text.setStyleSheet("background-color: transparent; font-size: 50px;")
        self.result_text.setReadOnly(True)
        self.result_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        right_layout.addWidget(self.result_text, alignment=QtCore.Qt.AlignTop)

        # 按钮布局
        button_layout = QHBoxLayout()

        self.btn_start_detection = QPushButton("开始", self.central_widget)
        self.btn_start_detection.setStyleSheet(
            "background-color: #4CAF50;"
            "color: white;"
            "border-radius: 8px;"
            "padding: 10px;"
        )
        self.btn_start_detection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_start_detection.clicked.connect(self.start_detection)
        button_layout.addWidget(self.btn_start_detection)

        self.btn_stop_detection = QPushButton("结束", self.central_widget)
        self.btn_stop_detection.setStyleSheet(
            "background-color: #FF3636;"
            "color: white;"
            "border-radius: 8px;"
            "padding: 10px;"
        )
        self.btn_stop_detection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_stop_detection.clicked.connect(sys.exit)
        button_layout.addWidget(self.btn_stop_detection)

        right_layout.addLayout(button_layout)
        layout.addLayout(right_layout)

        self.setWindowFlag(Qt.FramelessWindowHint)

        self.is_detection_started = False
        self.detection_frames = 0
        self.object_tracker = {}  # 物体跟踪器字典
        self.current_object_count = {}  # 当前帧物体数量

    def detect_objects_in_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(image).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pre = model.model(img, augment=False)
        pred = non_max_suppression(pre, conf_thres, iou_thres, None, False, max_det=10)

        detection_info = ""

        self.current_object_count = {}

        for det in pred[0]:
            xyxy = (det[0], det[1], det[2], det[3])
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)

            if det[4] >= conf_thres:
                # 绘制检测框
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                # 在中心点绘制圆形标记
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                class_label = int(det[5])
                class_name = names[class_label]
                confidence = det[4]

                # 显示类别和置信度信息
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 在中心点显示中心点坐标信息
                center_text = f" ({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if class_name not in self.current_object_count:
                    self.current_object_count[class_name] = 0
                self.current_object_count[class_name] += 1

        return frame, detection_info

    def detect_objects(self):
        if self.is_detection_started:
            ret, frame_camera = self.cap.read()
            ret, frame_mp4 = self.cap_mp4.read()

            if ret:
                frame_camera = cv2.resize(frame_camera, (640, 480))
                frame_mp4 = cv2.resize(frame_mp4, (640, 480))
                frame_camera, detection_info_camera = self.detect_objects_in_frame(frame_camera)

                qimage_camera = QImage(frame_camera.data, frame_camera.shape[1], frame_camera.shape[0],
                                       QImage.Format_BGR888)
                self.label_camera.setPixmap(QPixmap.fromImage(qimage_camera))
                self.label_camera.objects = detection_info_camera

                qimage_mp4 = QImage(frame_mp4.data, frame_mp4.shape[1], frame_mp4.shape[0],
                                    QImage.Format_BGR888)
                self.label_mp4.setPixmap(QPixmap.fromImage(qimage_mp4))

                # 更新检测结果文本框的内容，包括物体数目
                class_counts_str = "\n".join(
                    [f"{classname}: {count}" for classname, count in self.current_object_count.items()])
                self.result_text.setText(
                    f"当前检测到的物体有：\n\n{class_counts_str}\n\n{detection_info_camera}")

        QTimer.singleShot(1, self.detect_objects)

    def start_detection(self):
        if not self.is_detection_started:
            self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
            self.cap_mp4 = cv2.VideoCapture("pt/垃圾分类宣传片.mp4")  # 打开MP4视频文件，替换为你的文件路径

            if not self.cap.isOpened() or not self.cap_mp4.isOpened():
                self.result_text.setText("无法打开摄像头或视频文件")
                return

            self.is_detection_started = True
            self.detect_objects()

            # 初始化物体数量
            self.current_object_count = {}
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:  # 按下Esc键退出程序
            sys.exit()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:  # 鼠标左键按下时拖动窗口
            self.dragPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
