import time
import cv2
import pyautogui
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from mss import mss
import pygetwindow as gw

class ScreenCaptureThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, bbox):
        super(ScreenCaptureThread, self).__init__()
        self.bbox = bbox

    def run(self):
        sct = mss()

        while True:
            screenshot = sct.grab(self.bbox)
            img = np.array(screenshot)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(512, 384, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)

    def update_bbox(self, bbox):
        self.bbox = bbox

class RecognitionThread(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        # 这里是识别操作的代码
        # 在每个关键步骤，发射一个信号来更新进度条
        for i in range(1, 101):
            time.sleep(0.1)  # 这里用sleep模拟耗时的操作
            self.progress_signal.emit(i)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PyQt5 Screen Capture")
        self.setGeometry(100, 100, 800, 600)

        # layout
        vbox = QVBoxLayout(self)

        # combobox for window selection
        self.combobox = QComboBox(self)
        self.combobox.addItems(gw.getAllTitles())
        self.combobox.currentIndexChanged.connect(self.select_window)
        vbox.addWidget(self.combobox)

        # label for showing screen capture
        self.label = QLabel(self)
        vbox.addWidget(self.label)

        # label for showing matched result
        self.result_label = QLabel(self)
        # load an image from file
        pixmap = QPixmap('image/UI/Start_UI/Image_Information_widget.png')
        # set the image to result_label
        self.result_label.setPixmap(pixmap)
        vbox.addWidget(self.result_label)

        # 创建一个新的QLabel用于显示匹配成功的信息
        self.info_label = QLabel(self)
        # 设置初始信息
        self.info_label.setText('等待匹配...')
        vbox.addWidget(self.info_label)

        # button for triggering mouse and keyboard actions
        self.btn = QPushButton('Trigger actions', self)
        self.btn.clicked.connect(self.trigger_actions)
        vbox.addWidget(self.btn)

        self.setLayout(vbox)

        # create thread with dummy bbox
        self.th = ScreenCaptureThread({'top': 0, 'left': 0, 'width': 1, 'height': 1})
        self.th.changePixmap.connect(self.setImage)
        self.th.start()

        # progress bar for showing recognition progress
        self.progress_bar = QProgressBar(self)
        vbox.addWidget(self.progress_bar)

        # recognition thread
        self.recognition_thread = RecognitionThread()
        self.recognition_thread.progress_signal.connect(self.update_progress_bar)

        # cancel button
        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.clicked.connect(self.cancel_recognition)
        vbox.addWidget(self.cancel_button)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def cancel_recognition(self):
        # terminate recognition thread
        self.recognition_thread.terminate()
        self.recognition_thread.wait()

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def trigger_actions(self):
        # start recognition thread
        self.recognition_thread.start()
        # 截图选中的窗口
        sct = mss()
        screenshot = np.array(sct.grab(self.th.bbox))

        # 将截图转为灰度图像
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # 加载模板图像
        template = cv2.imread('image/UI/Start_UI/Social_media.png', 0)

        # 模板匹配
        res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 如果最大匹配值大于阈值，那么我们认为找到了匹配的模板
        threshold = 0.5
        if max_val > threshold:
            # 计算模板的位置
            w, h = template.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # 在截图上画一个矩形来标记找到的位置
            cv2.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 2)

            # 将结果转为QImage格式并显示在result_label上
            img_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(512, 384, Qt.KeepAspectRatio)
            self.result_label.setPixmap(QPixmap.fromImage(p))

            # 显示匹配成功的信息
            self.info_label.setText('匹配成功！')
        else:
            # 如果最大匹配值不大于阈值，那么我们认为没有找到匹配的模板
            self.info_label.setText('没有找到匹配的内容！')

    def select_window(self):
        win_title = self.combobox.currentText()
        win = gw.getWindowsWithTitle(win_title)[0]
        bbox = {'top': win.top, 'left': win.left, 'width': win.width, 'height': win.height}
        self.th.update_bbox(bbox)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
