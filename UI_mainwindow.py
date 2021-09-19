import os
import ctypes
import qtawesome as qtawesome
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QCursor,  QFont, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import tkinter as tk
import sys
import recognition_main
from UI_information import ImageViewer_1
from UI_Violation import ImageViewer_2

from PyQt5.QtWidgets import QPushButton, QDialog, QTextEdit, QFileDialog, QScrollArea, QWidget, \
    QLabel


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


class mywindow(QtWidgets.QMainWindow, Public):
    windowMinimumed = pyqtSignal()
    windowMaximumed = pyqtSignal()

    def __init__(self):
        super(mywindow, self).__init__()
        self.thread_pool = recognition_main.RecognitionThreadPool()
        self.video_names = []
        self.show_frame_bool = False
        self.update_list = False
        self.init_ui()

    def init_ui(self):
        # right_list = pyqtSignal()
        # Index_Signle = pyqtSignal(int)

        # 一、整体设置
        self.setFixedSize(1400, 900)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        self.main_widget.setStyleSheet("background-color: rgb(41, 36, 33);")  # 设置整体背景颜色

        self.left_widget = QtWidgets.QWidget()  # 创建左侧播放栏
        self.left_widget.setObjectName("视频栏")
        self.left_layout = QtWidgets.QGridLayout()  # 创建左部件的网格布局
        self.left_widget.setLayout(self.left_layout)  # 设置左侧播放栏布局为网格布局
        # self.left_widget.setStyleSheet("background-color: rgb(192, 192, 192);")  # 设置左窗口颜色

        self.right_widget = QtWidgets.QWidget()  # 创建右侧播放栏
        self.right_widget.setObjectName("播放栏")
        self.right_layout = QtWidgets.QGridLayout()  # 创建右部件的网格布局
        self.right_widget.setLayout(self.right_layout)  # 设置右侧播放栏布局为网格布局
        self.right_widget.setStyleSheet("background-color: rgb(41, 36, 33);")  # 设置右窗口颜色

        self.bottom_widget = QtWidgets.QWidget()  # 创建底部控制栏
        self.bottom_widget.setObjectName("控制栏")
        self.bottom_layout = QtWidgets.QGridLayout()  # 创建底部件的网格布局
        self.bottom_widget.setLayout(self.bottom_layout)  # 设置底部控制栏布局为网格布局
        self.bottom_widget.setStyleSheet("background-color: rgb(41, 36, 33);")  # 设置底部颜色

        # 整体布局
        self.main_layout.addWidget(self.left_widget, 0, 0, 9, 8)  # 右侧部件在第0行第0列，占8行7列
        self.main_layout.addWidget(self.right_widget, 0, 8, 9, 3)  # 右侧部件在第0行第7列，占8行2列
        self.main_layout.addWidget(self.bottom_widget, 9, 0, 3, 11)  # 底侧部件在第0行第7列，占8行2列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        # 二、左边布局的控件内容
        # 1.播放器窗口
        self.Videowindow = QVideoWidget(self)  # 创建播放器窗口
        self.mplayer = QMediaPlayer(self)  # 创建播放器
        self.mplayer.setVideoOutput(self.Videowindow)  # nplayer设置视频输出窗体（QVideoWideget）
        # self.Videowindow.show()
        # self.mplayer.play()

        # 2.进度条
        self.left_process_bar = QtWidgets.QProgressBar()  # 播放进度部件
        self.left_process_bar.show()
        self.left_process_bar.setFixedHeight(4)  # 设置进度条高度
        self.left_process_bar.setTextVisible(False)  # 不显示进度条文字
        self.mplayer.positionChanged.connect(self.PlaySlide)
        self.mplayer.durationChanged.connect(self.MediaTime)

        # 3.播放控件
        self.left_playconsole_widget = QtWidgets.QWidget()  # 播放控制部件
        self.left_playconsole_layout = QtWidgets.QGridLayout()  # 播放控制部件网格布局层
        self.left_playconsole_widget.setLayout(self.left_playconsole_layout)

        self.console_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.chevron-left', color='#292421'), "")
        self.console_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.chevron-right', color='#292421'), "")
        self.console_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.pause', color='#292421', font=18), "")
        self.console_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.play', color='#292421', font=18), "")
        self.console_button_3.setIconSize(QtCore.QSize(30, 30))
        self.console_button_4.setIconSize(QtCore.QSize(30, 30))
        self.console_button_4.clicked.connect(self.PlayVideo)
        self.console_button_3.clicked.connect(self.StopVideo)

        self.left_playconsole_layout.addWidget(self.console_button_1, 0, 0)
        self.left_playconsole_layout.addWidget(self.console_button_2, 0, 3)
        self.left_playconsole_layout.addWidget(self.console_button_3, 0, 2)
        self.left_playconsole_layout.addWidget(self.console_button_4, 0, 1)
        self.left_playconsole_layout.setAlignment(QtCore.Qt.AlignCenter)  # 设置布局内部件居中显示

        # 左上标识
        self.title = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " TRY TRY TRY 交通场景智能识别系统")
        self.title.setStyleSheet("color:rgb(192, 192, 192);font-size:25px;text-align: left;border:none;")
        self.title.setFont(QFont("Microsoft YaHei"))

        # 添加布局
        self.left_vedio = QtWidgets.QWidget()
        self.left_vedio.setObjectName("视频")
        self.left_layout_vedio = QtWidgets.QGridLayout()
        self.left_vedio.setLayout(self.left_layout_vedio)
        self.left_vedio.setStyleSheet("background-color: rgb(192, 192, 192);")

        # 设置左边layout
        self.left_layout_vedio.addWidget(self.Videowindow, 0, 0, 8, 10)
        self.left_layout_vedio.addWidget(self.left_process_bar, 9, 0, 1, 10)
        self.left_layout_vedio.addWidget(self.left_playconsole_widget, 10, 0, 1, 10)
        self.left_layout.addWidget(self.title, 0, 0, 1, 10)
        self.left_layout.addWidget(self.left_vedio, 2, 0, 10, 10)

        # 三、右边布局的控件内容
        # 页面控制按钮
        self.right_wcontrol_widget = QtWidgets.QWidget()  # 页面控制部件
        self.right_wcontrol_layout = QtWidgets.QGridLayout()  # 页面控制部件网格布局层
        self.right_wcontrol_widget.setLayout(self.right_wcontrol_layout)
        self.right_close = QtWidgets.QPushButton(qtawesome.icon('fa.window-close', color='#292421'), '')  # 关闭按钮
        self.right_visit = QtWidgets.QPushButton(qtawesome.icon('fa.window-maximize', color='#292421'), '')  # 全屏按钮
        self.right_mini = QtWidgets.QPushButton(qtawesome.icon('fa.window-minimize', color='#292421'), '')  # 最小化按钮
        self.right_wcontrol_layout.addWidget(self.right_close, 0, 2)
        self.right_wcontrol_layout.addWidget(self.right_mini, 0, 0)
        self.right_wcontrol_layout.addWidget(self.right_visit, 0, 1)
        self.right_close.clicked.connect(self.quit)  # 关闭按钮设置
        # self.right_close.clicked.connect(lambda: exit(0))
        self.right_mini.clicked.connect(self.showMinimized)
        self.right_close.resize(self.right_close.sizeHint())
        self.right_visit.clicked.connect(self.slot_max_or_recv)

        # 播放列表相关
        self.right_list1_widget = QtWidgets.QWidget()  # 列表控制部件2(不包含播放列表文字)
        self.right_list1_layout = QtWidgets.QGridLayout()  # 列表控制部件网格布局层2
        self.right_list1_widget.setLayout(self.right_list1_layout)

        self.right_label_1 = QtWidgets.QPushButton("————————播放列表————————")  # 播放列表上划线
        self.right_label_1.setObjectName('right_label')
        self.right_label_5 = QtWidgets.QPushButton("————————————————————")  # 播放列表下划线
        self.right_label_5.setObjectName('right_label_5')

        # 右侧播放列表按钮
        self.scroll_contents = QWidget()
        self.scroll_contents.setGeometry(0, 0, 0, 0)
        self.scroll_contents.setMinimumSize(100, 1000)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.scroll_contents)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.scroll_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.scroll_layout.setAlignment(QtCore.Qt.AlignTop)
        self.scroll_contents.setLayout(self.scroll_layout)  # 设置窗口主部件布局为网格布局

        '''
        for c in range(22):
            self.number = QLabel(self.scroll_contents)
            self.number.setText(str(c + 1))
            self.number.setStyleSheet("color: rgb(192, 192, 192);")
            self.number.move(5, (c * 45)+6)



            self.right_button_1 = QPushButton(self.scroll_contents)
            self.right_button_1.setText('播放视频')
            self.right_button_1.setStyleSheet("background-color: rgb(192, 192, 192);color:black")
            self.right_button_1.move(40, c * 45)
            self.right_button_2 = QPushButton(self.scroll_contents)
            self.right_button_2.setText('查看信息')
            self.right_button_2.setStyleSheet("background-color: rgb(192, 192, 192);color:black")
            self.right_button_2.move(180, c * 45)
            '''

        self.right_list1_layout.addWidget(self.scroll_area, 0, 0)

        '''
        #视频列表文字信息（无用）
        self.right_label_1_1 = QtWidgets.QPushButton("视频名称：vedio-01\n总时长：0h/0m/43s/80ms \nFPS：25\n总通过数：25")  # 播放列表文字
        self.right_label_1_1.setObjectName('right_label_1_1')
        self.right_label_1_1.setStyleSheet("border:none;text-align:left;")
        self.right_label_1_2 = QtWidgets.QPushButton("——————————————————")      #播放列表文字
        self.right_label_1_2.setObjectName('right_label_1_2')
        self.right_label_1_2.setStyleSheet("border:none;")
        self.right_label_2_1 = QtWidgets.QPushButton("视频名称：vedio-02\n总时长：0h/0m/43s/80ms \nFPS：25\n总通过数：25")  # 播放列表文字
        self.right_label_2_1.setObjectName('right_label_2_1')
        self.right_label_2_1.setStyleSheet("border:none;text-align:left;")
        self.right_label_2_2 = QtWidgets.QPushButton("——————————————————")      #播放列表文字
        self.right_label_2_2.setObjectName('right_label_2_2')
        self.right_label_2_2.setStyleSheet("border:none;")
        self.right_label_3_1 = QtWidgets.QPushButton("视频名称：vedio-03\n总时长：0h/0m/43s/80ms \nFPS：25\n总通过数：25")  # 播放列表文字
        self.right_label_3_1.setObjectName('right_label_3_1')
        self.right_label_3_1.setStyleSheet("border:none;text-align:left;")
        self.right_label_3_2 = QtWidgets.QPushButton("——————————————————")      #播放列表文字
        self.right_label_3_2.setObjectName('right_label_3_2')
        self.right_label_3_2.setStyleSheet("border:none;")
        self.right_label_4_1 = QtWidgets.QPushButton("视频名称：vedio-04\n总时长：0h/0m/43s/80ms \nFPS：25\n总通过数：25")  # 播放列表文字
        self.right_label_4_1.setObjectName('right_label_4_1')
        self.right_label_4_1.setStyleSheet("border:none;text-align:left;")
        '''

        self.mplayList = QMediaPlaylist()  # 创建播放列表
        # self.mplayList.addMedia(QMediaContent(QUrl.fromLocalFile("D:\学习\大二\软件杯\MAH00057\MAH00057-result.avi")))
        # self.mplayList.addMedia(
        #    QMediaContent(QUrl.fromLocalFile("D:\学习\大二\大二下/COMP2007J  计算机组成原理\课程视频\Lecture1-introduction.mp4")))
        # self.mplayList.addMedia(QMediaContent(QUrl.fromLocalFile("视频地址")))
        # self.mplayList.addMedia(QMediaContent(QUrl.fromLocalFile("视频地址")))
        self.mplayer.setPlaylist(self.mplayList)
        self.mplayList.setPlaybackMode(QMediaPlaylist.Sequential)
        self.mplayList.setCurrentIndex(0)
        self.video_index = 0
        self.mplayer.play()

        # 左侧视频控制按钮
        self.console_button_1.clicked.connect(self.video_play_previous)
        self.console_button_2.clicked.connect(self.video_play_next)

        # list尝试
        # self.right_list = QtWidgets.QListWidget  # 列表控制部件2
        # self.right_list.itemDoubleClicked.connect(self.GetItem)
        # self.right_list1_widget.addItem("1")
        # self.right_list1_widget.addItem("2")

        '''
            #list播放按钮
        self.right_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " 1")   # 创建播放按钮1
        self.right_button_1.setObjectName('right_button')
        self.right_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " 2")   # 创建播放按钮2
        self.right_button_2.setObjectName('right_button')
        self.right_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " 3")   # 创建播放按钮3
        self.right_button_3.setObjectName('right_button')
        self.right_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.car', color='white'), " 4")   # 创建播放按钮4
        self.right_button_4.setObjectName('right_button')

        #视频列表整体布局
        self.right_list1_layout.addWidget(self.right_label_1_1, 0, 0)
        self.right_list1_layout.addWidget(self.right_button_1, 0, 1)
        self.right_list1_layout.addWidget(self.right_label_1_2, 1, 0, 1, 2)

        self.right_list1_layout.addWidget(self.right_label_2_1, 2, 0)
        self.right_list1_layout.addWidget(self.right_button_2, 2, 1)
        self.right_list1_layout.addWidget(self.right_label_2_2, 3, 0, 1, 2)

        self.right_list1_layout.addWidget(self.right_label_3_1, 4, 0)
        self.right_list1_layout.addWidget(self.right_button_3, 4, 1)
        self.right_list1_layout.addWidget(self.right_label_3_2, 5, 0, 1, 2)

        self.right_list1_layout.addWidget(self.right_label_4_1, 6, 0)
        self.right_list1_layout.addWidget(self.right_button_4, 6, 1)
        self.right_list1_layout.setAlignment(QtCore.Qt.AlignCenter)  # 设置布局内部件居中显示

        '''

        '''
        self.right_button_1.clicked.connect(lambda: self.mplayList.setCurrentIndex(0))
        self.right_button_2.clicked.connect(lambda: self.mplayList.setCurrentIndex(1))
        self.right_button_3.clicked.connect(lambda: self.mplayList.setCurrentIndex(2))
        self.right_button_4.clicked.connect(lambda: self.mplayList.setCurrentIndex(3))
        '''

        #  右侧控件位置
        self.right_layout.addWidget(self.right_wcontrol_widget, 0, 8, 1, 3)
        self.right_layout.addWidget(self.right_label_1, 1, 8, 1, 3)  # 设置播放列表txt位置
        self.right_layout.addWidget(self.right_list1_widget, 2, 8, 4, 3)
        self.right_layout.addWidget(self.right_label_5, 7, 8, 1, 3)

        # 右侧控件美化
        self.right_close.setFixedSize(40, 40)  # 设置关闭按钮的大小
        self.right_visit.setFixedSize(40, 40)  # 设置按钮大小
        self.right_mini.setFixedSize(40, 40)  # 设置最小化按钮大小
        self.right_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.right_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.right_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.right_label_1.setStyleSheet("QPushButton{border:none;color:white;}")
        self.right_label_5.setStyleSheet("QPushButton{border:none;color:white;}")
        self.right_list1_widget.setStyleSheet("QPushButton{color:white;}")
        '''
        self.right_button_1.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.right_button_2.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.right_button_3.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.right_button_4.setStyleSheet("background-color: rgb(192, 192, 192);")
        '''

        # 四、底部布局的控件内容
        # 底部区域划分
        self.bottom_button_widget = QtWidgets.QWidget()  # 按钮区域
        self.bottom_button_layout = QtWidgets.QGridLayout()
        self.bottom_button_widget.setLayout(self.bottom_button_layout)
        self.bottom_text_widget = QtWidgets.QWidget()  # 按钮区域
        self.bottom_text_layout = QtWidgets.QGridLayout()
        self.bottom_text_widget.setLayout(self.bottom_text_layout)

        # 底部按钮
        self.bottom_button_1 = QtWidgets.QPushButton("导入视频")
        self.bottom_button_1.setObjectName('bottom_button_1')
        self.bottom_button_1.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.bottom_button_2 = QtWidgets.QPushButton("显示过程")
        self.bottom_button_2.setObjectName('bottom_button_2')
        self.bottom_button_2.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.bottom_button_layout.addWidget(self.bottom_button_1, 0, 0)
        self.bottom_button_layout.addWidget(self.bottom_button_2, 1, 0)

        # 底部文本框
        self.textEdit = QTextEdit()
        self.textEdit.setPlaceholderText('请点击左侧导入视频文件')
        self.textEdit.setStyleSheet("background-color: rgb(245,255,250);border-radius:10px;")
        self.bottom_text_layout.addWidget(self.textEdit, 0, 0)

        self.textEdit_2 = QTextEdit()
        self.textEdit_2.setStyleSheet("background-color: rgb(245,255,250);border-radius:10px;")
        self.bottom_text_layout.addWidget(self.textEdit_2, 0, 1)

        # 添加控件至布局
        self.bottom_layout.addWidget(self.bottom_button_widget, 0, 0, 1, 1)
        self.bottom_layout.addWidget(self.bottom_text_widget, 0, 1, 1, 1)

        # 五、整体美化
        self.main_widget.setStyleSheet("background-color: rgb(41, 36, 33);")
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.setWindowOpacity(1)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        # 六、导入视频的按钮和显示过程的按钮
        self.bottom_button_1.clicked.connect(self.ImportVideo)
        self.bottom_button_2.clicked.connect(self.show_frame)
        self.count = -1

        self.tiemer1 = QTimer()
        self.tiemer2 = QTimer()
        self.tiemer1.timeout.connect(self.run)
        # self.judge = True
        # if self.judge == True:
        # self.judge_list = []
        # for i in range(22):
        # self.judge_list.append(False)
        self.tiemer1.timeout.connect(self.make_list)
        self.tiemer1.start(500)
        self.tiemer2.start(1000)

        '''
        count = -1
        if self.bottom_button_1:
            "1、点击导入视频按钮后，后端开始分析视频"
            self.textEdit.insertPlainText("正在分析视频...\n")
            count = count + 1

            print(self.openfile())
            if self.thread_pool.has_finished(self.openfile()[0]):
                "2、视频分析成功后，后端生成对应视频与文本，前端生成播放别表按钮"
                self.textEdit.insertPlainText("视频分析成功，请点击对应播放列表按钮\n")
                # 生成视频播放按钮
                self.right_button_1 = QPushButton(self.scroll_contents)
                self.right_button_1.setText('播放视频')
                self.right_button_1.setStyleSheet("background-color: rgb(192, 192, 192);color:black")
                self.right_button_1.move(40, count * 35)
                # 生成获取信息按钮
                self.right_button_2 = QPushButton(self.scroll_contents)
                self.right_button_2.setText('获取信息')
                self.right_button_2.setStyleSheet("background-color: rgb(192, 192, 192);color:black")
                self.right_button_2.move(180, count * 35)
                self.right_button_2.clicked.connect(self.onclick())
                '''

    def quit(self):
        self.thread_pool.stop_all()
        QCoreApplication.instance().quit()

    def video_play(self, index):
        index = int(index)
        if index > self.count:
            return
        self.mplayList.setCurrentIndex(index)
        '''
        diff = index - self.video_index
        if diff >= 0:
            for i in range(diff):
                self.mplayList.next()
        else:
            diff = -diff
            for i in range(diff):
                self.mplayList.previous()
                '''
        self.video_index = index

    def video_play_previous(self):
        self.mplayList.previous()
        self.video_index -= 1
        if self.video_index < 0:
            self.video_index = 0
        elif self.video_index > self.count:
            self.video_index = self.count

    def video_play_next(self):
        self.mplayList.next()
        self.video_index += 1
        if self.video_index < 0:
            self.video_index = 0
        elif self.video_index > self.count:
            self.video_index = self.count

    def show_frame(self):
        # print(self.show_frame_bool)
        if self.show_frame_bool:
            self.show_frame_bool = False
            self.thread_pool.show_frame(False)
        else:
            self.show_frame_bool = True
            self.thread_pool.show_frame(True)

    def make_list(self):
        # print(self.video_names)
        for name in self.video_names:
            # print(self.video_names)
            # print(self.thread_pool.has_finished(name))
            try:
                # if self.thread_pool.has_finished(name):
                # print(self.update_list)
                if self.update_list and self.thread_pool.has_finished(name) and not self.thread_pool.pool[name].in_list:
                    "2、视频分析成功后，后端生成对应视频与文本，前端生成播放别表按钮"
                    # 生成视频播放按钮
                    # if self.judge_list[self.count] != True:
                    self.count += 1
                    print(self.count)
                    self.d = locals()
                    index = name.find('.')
                    path = name[0: index]
                    _, file_name = os.path.split(name)
                    index = file_name.find('.')
                    path = '.\\' + file_name[0: index] + '\\' + file_name[0: index] + '-result.avi'
                    count = self.count
                    # print(path)
                    # path = 'D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_New\\video - 2\\video - 2 - result.avi'
                    self.mplayList.addMedia(QMediaContent(QUrl.fromLocalFile(path)))

                    self.d["title" + str(self.count)] = QLabel(self.scroll_contents)
                    self.d["title" + str(self.count)].setText(file_name)
                    # self.d["title" + str(self.count)].setStyleSheet("border:none;")
                    self.d["title" + str(self.count)].setStyleSheet("color:white")
                    self.scroll_layout.addWidget(self.d["title" + str(self.count)], 2 * self.count, 1, 1, 4)

                    self.d["right_button_1_" + str(self.count)] = QPushButton(self.scroll_contents)
                    # self.d["right_button_1_" + str(self.count)].setObjectName(str(self.count))
                    # self.right_button_1 = QPushButton(self.scroll_contents)
                    self.d["right_button_1_" + str(self.count)].setText('播放视频')
                    self.d["right_button_1_" + str(self.count)].setStyleSheet(
                        "background-color: rgb(192, 192, 192);color:black")
                    self.scroll_layout.addWidget(self.d["right_button_1_" + str(self.count)], 2 * self.count + 1, 1, 1,
                                                 1)
                    # bug 在这里在这里
                    self.d["right_button_1_" + str(self.count)].clicked.connect(lambda: self.video_play(count))
                    self.mplayer.play()

                    # 生成获取信息按钮
                    self.d["right_button_2_" + str(self.count)] = QPushButton(self.scroll_contents)
                    self.d["right_button_2_" + str(self.count)].setText('获取信息')
                    self.d["right_button_2_" + str(self.count)].setStyleSheet(
                        "background-color: rgb(192, 192, 192);color:black")
                    self.scroll_layout.addWidget(self.d["right_button_2_" + str(self.count)], 2 * self.count + 1, 2, 1,
                                                 1)
                    self.d["right_button_2_" + str(self.count)].clicked.connect(
                        lambda: self.show_information(file_name[0: index]))

                    self.d["right_button_3_" + str(self.count)] = QPushButton(self.scroll_contents)
                    self.d["right_button_3_" + str(self.count)].setText('查看违章')
                    self.d["right_button_3_" + str(self.count)].setStyleSheet(
                        "background-color: rgb(192, 192, 192);color:black")
                    self.scroll_layout.addWidget(self.d["right_button_3_" + str(self.count)], 2 * self.count + 1, 3, 1,
                                                 1)
                    self.d["right_button_3_" + str(self.count)].clicked.connect(
                        lambda: self.show_violation(file_name[0: index]))

                    self.number = QLabel(self.scroll_contents)
                    self.number.setText(str(self.count + 1))
                    self.number.setStyleSheet("color: rgb(192, 192, 192);")
                    self.scroll_layout.addWidget(self.number, 2 * self.count, 0, 1, 1)

                    self.update_list = False
                    self.thread_pool.pool[name].in_list = True
                    break
                    # self.scroll_area.setWidget(self.right_button_2)
                    # self.right_button_2.clicked.connect(self.onclick())
                    # print("yeah")
            except Exception as e:
                print(e)
                pass

    def run(self):
        self.textEdit.clear()
        try:
            for name in self.video_names:
                if self.thread_pool.has_finished(name):
                    "2、视频分析成功后，后端生成对应视频与文本，前端生成播放别表按钮"
                    _, file_name = os.path.split(name)
                    index = file_name.find('.')
                    path = '.\\' + file_name[0: index] + '\\' + 'video_info.txt'
                    info_file = open(path, 'r')
                    # msg = info_file.read()
                    string = "视频%s\n分析成功，请点击对应播放列表按钮\n" % name
                    self.textEdit.insertPlainText(string)
                    if not self.thread_pool.pool[name].msg_printed:
                        msg = ''
                        while True:
                            line = info_file.readline()
                            if line[:2] == 'id':
                                break
                            else:
                                msg += line
                        msg += '\n'
                        info_file.close()
                        self.textEdit_2.insertPlainText(msg)
                        self.thread_pool.pool[name].msg_printed = True
                else:
                    process, percentage = self.thread_pool.get_process(name)
                    string = "视频%s\n正在分析:%s, %s%%\n" % (name, process, percentage)
                    # print(string)
                    self.textEdit.insertPlainText(string)
            '''
            for each in msgs:
                print(count, each)
                self.textEdit.insertPlainText("%d. " % count + each)
                count += 1
                pass
            '''
            # self.mplayer.play()
        except Exception as e:
            print(e)
            pass

    def ImportVideo(self):
        file = self.openfile()
        # print(file)
        if file[0] == '':
            return
        "1、点击导入视频按钮后，后端开始分析视频"
        self.textEdit.insertPlainText("正在分析视频...\n")
        try:
            setting_list = []
            VideoSelectionWindow(setting_list)
            # print(setting_list)
            self.thread_pool.add_video(file[0], camera_mod=setting_list[0], boundary_mod=setting_list[1],
                                       size_mod=setting_list[2], flow_detect=bool(setting_list[3]),
                                       direction_detect=bool(setting_list[4]), speed_detect=bool(setting_list[5]),
                                       front_end=self)
            self.video_names.append(file[0])
            self.thread_pool.pool[file[0]].in_list = False
            self.thread_pool.pool[file[0]].msg_printed = False
        except Exception as e:
            print(e)

    def PlaySlide(self, val):
        self.left_process_bar.setValue(int(val / 1000))

    def MediaTime(self, time):
        self.left_process_bar.setValue(0)
        self.time = self.mplayer.duration() / 1000
        self.left_process_bar.setRange(0, int(self.time))

    def PlayVideo(self):
        self.mplayer.play()

    def StopVideo(self):
        self.mplayer.pause()

    def openfile(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件')
        # 选好视频之后直接对线程池添加视频就会开始分析，同时前台其它的东西都可以正常运作不受印象
        # 查看是否完成和进度参见recognition_main那个文件
        # self.thread_pool.add_video(openfile_name[0])
        # print(openfile_name)
        return openfile_name

    def show_information(self, video_name: str):
        self.newWindow_1 = ImageViewer_1(video_name)
        self.newWindow_1.show()

    def show_violation(self, video_name: str):
        self.newWindow_2 = ImageViewer_2(video_name)
        self.newWindow_2.show()

    def slot_max_or_recv(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()


# 这个是导入视频前的选项窗口
class VideoSelectionWindow:

    def __init__(self, result_list: []):
        self.window = tk.Tk()
        self.window.title('识别选项')

        winapi = ctypes.windll.user32
        x = winapi.GetSystemMetrics(0)
        y = winapi.GetSystemMetrics(1)
        # print(x, y)
        m = x // 1920
        m = int(680 * m)

        self.window.geometry('%dx%d+500+250' % (m, int(m * 1)))
        self.result_list = result_list

        self.camera_chosen = False
        self.boundary_chosen = False
        self.size_chosen = False
        self.flow_chosen = False
        self.direction_chosen = False
        self.speed_chosen = False

        self.camera_var = tk.StringVar()
        self.camera_var.set(0)
        self.camera_label = tk.Label(self.window, width=60, text='选择相机模式', fg="red", bg="lightblue")
        self.camera_label.pack()
        self.camera_0 = tk.Radiobutton(self.window, text='模式0（推荐），相机俯角设置为30度，用于测速，注意此值仅为参考值！', variable=self.camera_var,
                                       value=0, command=self.print_camera_selection)
        self.camera_0.pack(anchor="w")
        self.camera_1 = tk.Radiobutton(self.window, text='模式1，相机俯角设置为45度，用于测速，注意此值仅为参考值！', variable=self.camera_var,
                                       value=1, command=self.print_camera_selection)
        self.camera_1.pack(anchor="w")

        self.boundary_var = tk.StringVar()
        self.boundary_var.set(0)
        self.boundary_label = tk.Label(self.window, width=60, text='选择区域设置', fg="red", bg="lightblue")  # QMA
        self.boundary_label.pack()
        self.boundary_0 = tk.Radiobutton(self.window, text='模式0（推荐），适合较近距离高俯角的情况，视野内只有行车道', variable=self.boundary_var,
                                         value=0,
                                         command=self.print_boundary_selection)
        self.boundary_0.pack(anchor="w")
        self.boundary_1 = tk.Radiobutton(self.window, text='模式1，适合视野较大、低俯角的情况，识别区域只包含路口中心', variable=self.boundary_var,
                                         value=1,
                                         command=self.print_boundary_selection)
        self.boundary_1.pack(anchor="w")

        self.size_var = tk.StringVar()
        self.size_var.set(0)
        self.size_label = tk.Label(self.window, width=60, text='选择跟踪大小', fg="red", bg="lightblue")  # QMA
        self.size_label.pack()
        self.size_0 = tk.Radiobutton(self.window, text='模式0（推荐），适合距离近，物体面积变化不大的情况', variable=self.size_var, value=0,
                                     command=self.print_size_selection)
        self.size_0.pack(anchor="w")
        self.size_1 = tk.Radiobutton(self.window, text='模式1，适合距离远，或物体大小变化比较明显的情况', variable=self.size_var, value=1,
                                     command=self.print_size_selection)
        self.size_1.pack(anchor="w")

        self.flow_var = tk.StringVar()
        self.flow_var.set(True)
        self.flow_label = tk.Label(self.window, width=60, text='流量监测（仅供参考）', fg="red", bg="lightblue")  # QMA
        self.flow_label.pack()
        self.flow_0 = tk.Radiobutton(self.window, text='开', variable=self.flow_var, value=True,
                                     command=self.flow_selection)
        self.flow_0.pack()
        self.flow_1 = tk.Radiobutton(self.window, text='关', variable=self.flow_var, value=False,
                                     command=self.flow_selection)
        self.flow_1.pack()

        self.direction_var = tk.StringVar()
        self.direction_var.set(True)
        self.direction_label = tk.Label(self.window, width=60, text='行车方向监测（仅供参考）', fg="red", bg="lightblue")  # QMA
        self.direction_label.pack()
        self.direction_0 = tk.Radiobutton(self.window, text='开', variable=self.direction_var, value=True,
                                          command=self.direction_selection)
        self.direction_0.pack()
        self.direction_1 = tk.Radiobutton(self.window, text='关', variable=self.direction_var, value=False,
                                          command=self.direction_selection)
        self.direction_1.pack()

        self.speed_var = tk.StringVar()
        self.speed_var.set(True)
        self.speed_label = tk.Label(self.window, width=60, text='速度监测（仅供参考）', fg="red", bg="lightblue")  # QMA
        self.speed_label.pack()
        self.speed_0 = tk.Radiobutton(self.window, text='开', variable=self.speed_var, value=True,
                                      command=self.speed_selection)
        self.speed_0.pack()
        self.speed_1 = tk.Radiobutton(self.window, text='关', variable=self.speed_var, value=False,
                                      command=self.speed_selection)
        self.speed_1.pack()
        self.confirm_button = tk.Button(self.window, text='确定', font=('Arial', 12), width=10, height=1,
                                        command=self.confirm)
        self.confirm_button.pack()

        self.window.mainloop()

    def print_camera_selection(self):
        self.camera_chosen = True

    def print_boundary_selection(self):
        self.boundary_chosen = True

    def print_size_selection(self):
        self.size_chosen = True

    def flow_selection(self):
        self.flow_chosen = True

    def direction_selection(self):
        self.direction_chosen = True

    def speed_selection(self):
        self.speed_chosen = True

    def confirm(self):
        self.result_list.append(int(self.camera_var.get()))
        self.result_list.append(int(self.boundary_var.get()))
        self.result_list.append(int(self.size_var.get()))
        self.result_list.append(int(self.flow_var.get()))
        self.result_list.append(int(self.direction_var.get()))
        self.result_list.append(int(self.speed_var.get()))
        self.window.destroy()
        '''
        # print(self.camera_var.get(), self.boundary_var.get(), self.size_var.get(), self.flow_var.get(), self.direction_var.get(), self.speed_var.get())
        if self.camera_chosen and self.boundary_chosen and self.size_chosen and self.flow_chosen and self.direction_chosen and self.speed_chosen:
            self.result_list.append(int(self.camera_var.get()))
            self.result_list.append(int(self.boundary_var.get()))
            self.result_list.append(int(self.size_var.get()))
            self.result_list.append(int(self.flow_var.get()))
            self.result_list.append(int(self.direction_var.get()))
            self.result_list.append(int(self.speed_var.get()))
            self.window.destroy()
        else:
            while len(self.result_list) > 0:
                self.result_list.pop()
            tk.messagebox.showerror('错误', '还有选项没有选！')
        '''


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(qtawesome.icon('fa.car', color='white')))
    gui = mywindow()
    gui.show()
    # while True:
    # time.sleep(0.2)
    # gui.update()
    sys.exit(app.exec_())

# 'recognition\\clpr_entry.py', 'recognition\\clpr_loation.py', 'recognition\\clpr_segmentation.py', 'recognition\\Items.py', 'recognition\\Video.py', 'tools\\Display.py', 'tools\\Geometry.py', 'tools\\'
