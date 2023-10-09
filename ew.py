from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QColor

app = QApplication([])

widget = QWidget()

layout = QVBoxLayout()
a='eee'
label = QLabel()
text = f"<font color='red'>{a}</font> <font color='blue'>World</font>"
label.setText(text)

# 设置第2行的字体颜色为绿色
palette = label.palette()
palette.setColor(label.foregroundRole(), QColor("green"))
label.setPalette(palette)

layout.addWidget(label)

widget.setLayout(layout)

widget.show()
app.exec()