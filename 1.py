from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Click Example")
        self.setGeometry(200, 200, 300, 200)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # 检查是否为左键点击
            print("Left button clicked")
        elif event.button() == Qt.RightButton:  # 检查是否为右键点击
            print("Right button clicked")

        position = event.pos()  # 获取鼠标点击位置
        print("Clicked position:", position.x(), position.y())


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()