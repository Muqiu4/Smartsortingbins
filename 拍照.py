import cv2
import os

dir = os.getcwd()
type = 'else waste'
count = 0
if not os.path.exists(type):
    os.mkdir(type)
path = os.path.join(dir, type)
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_x, start_y = -1, -1
end_x, end_y = -1, -1


def take_photo(frame):
    # 打开摄像头
    global path, count, type, cap

    # 读取摄像头的一帧
    name = type + str(count) + '.jpg'
    # 保存图片到指定文件夹
    file_path = os.path.join(path, name)
    cv2.imwrite(file_path, frame)
    print("照片已保存至", file_path)
    count += 1

def draw_rectangle(event, x, y, flags, param):
    # 鼠标事件回调函数
    global start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y


# 创建一个窗口，并绑定鼠标事件回调函数
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

while True:
    # 等待按键操作
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 675))
    if start_x>=0:
        cv2.rectangle(frame, (start_x-1, start_y-1), (start_x+641, start_y+481), (0, 255, 0), 1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    #print(start_x, start_y)
    # 按下空格键，拍照并退出循环
    if key == ord('q'):
        frame = frame[start_y:start_y + 480,start_x:start_x + 640]
        for i in range(5):
            take_photo(frame)

    # 按下ESC键，退出循环
    if key == ord('w'):
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
