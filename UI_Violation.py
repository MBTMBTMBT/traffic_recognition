import PyQt5.sip
import os
import re
import sys
import qtawesome
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QTextEdit, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QCursor, QFont


class Public(QDialog):
    """鼠标拖动窗口操作"""

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.m_flag = True
                self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
                event.accept()
                self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标
        except:
            pass

    def mouseMoveEvent(self, QMouseEvent):
        try:
            if Qt.LeftButton and self.m_flag:
                self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
                QMouseEvent.accept()
        except:
            pass

    def mouseReleaseEvent(self, QMouseEvent):
        try:
            self.m_flag = False
            self.setCursor(QCursor(Qt.ArrowCursor))
        except:
            pass


class ImageViewer_2(QtWidgets.QMainWindow, Public):
    def __init__(self, video_name):
        super(ImageViewer_2, self).__init__()
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.label = QLabel(self)
        self.textEdit = QTextEdit(self)
        self.listWidget = QtWidgets.QListWidget()
        self.pushButton_2 = QtWidgets.QPushButton("修改信息")
        self.video_name = video_name
        self.init_ui()

    def init_ui(self):

        self.right_widget = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)
        self.left_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        # 一、整体设置
        self.setFixedSize(1000, 800)
        self.move(300, 200)
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        self.main_widget.setStyleSheet("background-color: rgb(41, 36, 33);")  # 设置整体背景颜色
        self.setWindowTitle("违规信息")
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.right_widget = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)
        self.left_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        self.main_layout.addWidget(self.left_widget, 0, 0, 20, 16)  # 右侧部件在第0行第0列，占8行7列
        self.main_layout.addWidget(self.right_widget, 0, 16, 20, 5)  # 右侧部件在第0行第0列，占8行7列

        # 二
        self.title = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " 违规信息")
        self.title.setStyleSheet("color:rgb(192, 192, 192);font-size:25px;text-align: left;border:none;")
        self.title.setFont(QFont("Microsoft YaHei"))
        self.left_layout.addWidget(self.title, 0, 0, 1, 15)

        self.listWidget.setObjectName("listWidget")
        self.listWidget.setStyleSheet("color: rgb(192, 192, 192);")
        self.listWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.right_layout.addWidget(self.listWidget, 1, 0, 20, 5)

        self.left_layout.addWidget(self.label, 1, 0, 480, 30)
        self.label.setStyleSheet("background-color: rgb(3,3,3);border-radius:10px;")  # 设置整体背景颜色
        self.listWidget.currentItemChanged.connect(self.image)  # 这是点击item会返回item的名称

        self.left_layout.addWidget(self.textEdit, 481, 0, 96, 30)
        self.textEdit.setStyleSheet("background-color: rgb(192,192,192);border-radius:10px;")  # 设置整体背景颜色

        self.left_layout.addWidget(self.pushButton_2, 575, 27)
        self.pushButton_2.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.pushButton_2.clicked.connect(self.openfile)

        # 三、播放控件

        self.top_wcontrol_widget = QtWidgets.QWidget()  # 页面控制部件
        self.top_wcontrol_layout = QtWidgets.QGridLayout()  # 页面控制部件网格布局层
        self.top_wcontrol_widget.setLayout(self.top_wcontrol_layout)
        self.right_close = QtWidgets.QPushButton(qtawesome.icon('fa.window-close', color='#292421'), '')  # 关闭按钮
        self.right_visit = QtWidgets.QPushButton(qtawesome.icon('fa.window-maximize', color='#292421'), '')  # 全屏按钮
        self.right_mini = QtWidgets.QPushButton(qtawesome.icon('fa.window-minimize', color='#292421'), '')  # 最小化按钮
        self.top_wcontrol_layout.addWidget(self.right_close, 0, 2)
        self.top_wcontrol_layout.addWidget(self.right_mini, 0, 0)
        self.top_wcontrol_layout.addWidget(self.right_visit, 0, 1)
        self.right_close.setFixedSize(35, 35)  # 设置关闭按钮的大小
        self.right_visit.setFixedSize(35, 35)  # 设置按钮大小
        self.right_mini.setFixedSize(35, 35)  # 设置最小化按钮大小
        self.right_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.right_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.right_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.right_layout.addWidget(self.top_wcontrol_widget, 0, 0, 1, 5)

        self.right_close.clicked.connect(self.close)  # 关闭按钮设置
        self.right_close.resize(self.right_close.sizeHint())
        self.right_mini.clicked.connect(self.showMinimized)
        self.right_close.resize(self.right_close.sizeHint())
        self.right_visit.clicked.connect(self.slot_max_or_recv)

        # 四、整体美化
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框

        try:
            self.viewlist()
        except Exception as e:
            print(e)

    def image(self):
        # print(self.listWidget.currentItem().text())
        self.imagefile = self.listWidget.currentItem().text()
        self.name = self.imagefile.replace('.jpg', '')
        self.name = self.name + '.txt'
        # print(self.imagefile)

        try:
            with open(self.name, 'r') as f:
                msg = f.read()
                identification = re.findall(r'id：([0-9][0-9]*)', msg)[0]
                self.textEdit.setPlainText(msg)
                msg += '\n'
                f.close()
        except Exception as e:
            print(e)
        try:
            with open(self.video_name + '\\id' + identification + '.txt') as f:
            # with open(self.video_name + '\\id' + identification + '.txt') as f:
                msg1 = f.read()
                self.textEdit.setPlainText(msg + msg1)
                f.close()
        except Exception as e:
            print(e)
        jpg = QtGui.QPixmap(self.imagefile).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    def openfile(self):
        try:
            os.startfile(str(self.name))
        except Exception as e:
            print(e)

    def viewlist(self):
        self.list = os.listdir('.\\' + self.video_name)
        # self.list = os.listdir("D:\学习\大二\软件杯\MAH00057")
        # self.list.sort(key=lambda x: (x[8]))
        # print(self.list)
        for i in self.list:
            if i.endswith('jpg'):
                # print(i)
                self.listWidget.addItem('.\\' + self.video_name + "/" + i)

    def slot_max_or_recv(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ImageViewer_2("video-2")
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
