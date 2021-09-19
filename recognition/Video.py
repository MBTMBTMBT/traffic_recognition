# coding=utf8
import cv2 as cv
import os
from enum import Enum
from math import *
from tools import Display, Geometry
from recognition import Items

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class Frame(object):

    # 定义每一帧参数；所属视频及出现时间
    def __init__(self, cv_frame, serial_num, video):
        self.cv_frame = cv_frame
        self.serial_num = serial_num
        self.video = video
        self.milliseconds = serial_num / video.fps * 1000

    def __str__(self):
        rst = "Frame num %d, %d, %s" % (self.serial_num, self.milliseconds, Display.format_time(self.milliseconds))
        return rst

    @staticmethod
    def position_p720_to_p4k(position: ()):
        x0 = int(position[0] - 640)
        y0 = int(position[1] - 360)
        x = int(1920 + x0 * 3)
        y = int(1080 + y0 * 3)
        return x, y

    @ staticmethod
    def position_p4k_to_p720(position: ()):
        x0 = int(position[0] - 1920)
        y0 = int(position[1] - 1080)
        x = int(640 + x0 // 3)
        y = int(360 + y0 // 3)
        return x, y

    @ staticmethod
    def position_pixel_transfer(position: (), origin: (), target: ()):
        # x0 = int(position[0] - origin[0] // 2)
        # y0 = int(position[1] - origin[1] // 2)
        x0 = position[0]
        y0 = position[1]
        # x = int(target[0] // 2 + x0 * (target[0] // origin[0]))
        # y = int(target[1] // 2 + y0 * (target[1] // origin[1]))
        x = int(x0 * (target[0] / origin[0]))
        y = int(y0 * (target[1] / origin[1]))
        return x, y


# -----------------------------------------
class Camera(object):

    # 定义相机相关参数：相机所处高度；水平、竖直视角范围及俯角
    def __init__(self, camera_height: float, visual_angle_vertical: int, visual_angle_horizontal: int, depression_angle: int):
        self.camera_height = camera_height
        # 将角度值转化成弧度制表示
        self.visual_angle_vertical = radians(visual_angle_vertical)
        self.visual_angle_horizontal = radians(visual_angle_horizontal)
        self.depression_angle = radians(depression_angle)
        self.vision_height = self.count_vision_height()

    # 通过计算相机所处高度及其视角范围计算拍摄画面中物体的实际高度
    def count_vision_height(self):
        h = self.camera_height
        a = self.visual_angle_vertical
        b = self.depression_angle
        return 2 * sin(b / 2) * h / sin(a + b / 2)

    # 通过相机参数及画面中物体与标准点距离的数值计算，得到实际物体与相机的水平距离
    def count_distance(self, x: float):
        h = self.camera_height
        a = self.visual_angle_vertical
        b = self.depression_angle
        m = sin(a + b / 2)
        # print(2 * ((h / m) ** 2), cos(radians(90) - (b / 2)))
        numerator = 2 * ((h / m) ** 2) - 2 * x * h * cos(radians(90) - b / 2) / m
        denominator = 2 * h / m * sqrt(x ** 2 + (h / m) ** 2 - 2 * x * h * cos(radians(90) - b / 2) / m)
        theta = a + b / 2 - acos(numerator / denominator)
        # print(numerator)
        # print(denominator)
        # print(theta)
        return h / tan(theta)

    # 计算物体在画面中的像素高度
    def count_relative_height(self, pixel: int, pixel_height=1080):
        return pixel / pixel_height * self.vision_height

    # 计算物体在画面中的斜边偏移量
    def count_horizontal_offset(self, distance, pixel, pixel_width=1920, pixel_height=1080):
        # hypotenuse = sqrt(self.camera_height ** 2 + distance ** 2)
        # width = 2 * tan(self.visual_angle_horizontal / 2) * hypotenuse
        # horizontal_offset = pixel / pixel_width * width - width / 2
        # return horizontal_offset
        y = distance
        Y = self.count_distance(pixel_height)  # 画面最远距离
        X = Y * tan(self.visual_angle_horizontal / 2)  # 最远的那个水平距离的一半
        a = ((pixel - pixel_width / 2) / (pixel_width / 2)) * X  # 处于最远距离的那个偏移量
        m = y * X / (2 * Y)
        rst = 2 * a * m / X
        return rst


# -----------------------------------------
class Video(object):

    # 生成最终指定视频文件
    def __init__(self, video_capture, video_name='untitled'):
        # 假如无法打开原视频文件，则报错
        if not video_capture.isOpened():
            raise OSError("文件打开失败！")
        # 设定生成视频参数：画面大小、帧率等
        self.video_capture = video_capture
        self.fps = int(video_capture.get(cv.CAP_PROP_FPS))
        self.picture_rect \
            = Geometry.Rect(0, 0, video_capture.get(cv.CAP_PROP_FRAME_WIDTH),
                            video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.total_frames_num = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        self.items = []
        self.died_items = []
        self.grave = []
        self.item_count = 0
        self.realtime = 0
        self.flow_record = []
        self.overload = False
        self.overload_record = []
        # size = (int(self.picture_rect.width), int(self.picture_rect.height))
        # 这个地方输出视频暂时保存为1920 1080
        size = (1920, 1080)
        _, video_name = os.path.split(video_name)
        index = video_name.find('.')
        # 其文件名为：“原视频文件名”+"-result.avi"
        if index == -1:
            result_name = video_name + "-result.avi"
        else:
            result_name = video_name[0:index] + "-result.avi"
        self.video_name = video_name[0:index]
        path = "." + os.sep + self.video_name
        try:
            os.mkdir(path)
        except WindowsError:
            pass
        # 指定生成视频格式
        self.video_write \
            = cv.VideoWriter('.\\%s\\' % self.video_name + result_name,
                             cv.VideoWriter_fourcc('M', 'P', '4', '2'), self.fps, size)
        # 'I', '4', '2', '0'
        # ’F’, ’L’, ’V’, ’1’
        # 'M', 'P', '4', '2'

    # 将当前状态下活跃的物体添加到视频列表中
    def add_item(self, item: Items.Item):
        self.items.append(item)

    # 向生成的视频中写入准备好的帧
    def add_frame_to_video(self, cv_frame):
        self.video_write.write(cv_frame)

    def __str__(self):
        return "frame num: %d, fps: %d, size: %d * %d" \
               % (self.total_frames_num, self.fps, self.picture_rect.width, self.picture_rect.height)

    # 在指定文件中存储视频参数：储存路径、名称、时长、总通过车辆数及被追踪物体参数
    def save_video_info(self):
        time_length = self.total_frames_num / self.fps * 1000
        time_length_ms = time_length
        time_length = Display.format_time(time_length)
        file = open('%s\\video_info.txt' % self.video_name, 'w')
        file.write("* 视频名；%s\n" % self.video_name)
        file.write("* 总时长；%s, FPS：%s\n" % (time_length, self.fps))
        file.write("* 总通过数：%d\n" % self.item_count)
        minutes = time_length_ms / 60000
        if minutes != 0:
            flow = self.item_count // minutes
        else:
            flow = 9999
        file.write("* 总通过密度：<= %d辆/分钟\n* 超限流量统计：\n" % flow)
        for each in self.overload_record:
            file.write("\t* 流量：%d辆/分钟，时间：%s\n" % (each[2], Display.format_time(each[3])))
        for each in self.grave:
            file.write("id%d\n" % each.identification)
        file.close()

    # 在指定文件中存储被追踪的出现并离开画面的移动物体(dead_item)参数：
    # 总数量、出现时间、离开时间、位移距离及平均速度
    def save_dead_item(self, item):
        # file = open('%s\\%d.txt' % (self.video_name, item.identification), 'w')
        self.item_count += 1
        count = 0
        for each_img in item.quick_shots:
            string = "%s\\%d_%d.png" % (self.video_name, item.identification, count)
            # cv.imwrite(string, each_img)
            cv.imencode('.png', each_img)[1].tofile(string)
            count += 1
        file = open('%s\\id%d.txt' % (self.video_name, item.identification), 'w')
        if item.get_type() == 1:
            file.write("* 汽车\n")
            file.write("* 预测车牌号：%s\n" % item.predicted_plate)
            file.write("* 出现时间；%dms %s\n" % (item.start_time, Display.format_time(item.start_time)))
            file.write("* 离开时间：%dms %s\n" % (item.end_time, Display.format_time(item.end_time)))
            start_point = item.real_trace[0]
            end_point = item.real_trace[-2]
            file.write("* 位移距离：%.2fm\n"
                       % sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2))
            file.write("* 平均速度：%.2fm/s <-> %.2fkm/h\n" % (item.average_speed, item.average_speed * 3.6))
            file.write("* 车牌识别结果统计：\n")
            for plate, num in item.plates.items():
                file.write("\t* %s - %d\n" % (plate, num))
        elif item.get_type() == 2:
            file.write("* 摩托/自行车\n")
        elif item.get_type() == 3:
            file.write("* 行人\n")
        file.close()
        try:
            file = open('%s\\id%d-plate\\plates.txt' % (self.video_name, item.identification), 'w')
            count = 0
            for plate_str in item.plate_strs:
                file.write("%d: %s\n" % (count, plate_str))
                count += 1
            file.close()
        except FileNotFoundError:
            pass
        # file.write("%f, %f; %f, %f\n" % (end_point[0], end_point[1], start_point[0], start_point[1]))

    # 对于每个出现在被追踪到的画面中并离开画面的移动物体，记录他们的各个参数
    # 为了多线程方便，此方法不再直接使用self.dead_items列表
    def save_dead_items(self, died_items: []):
        for each in died_items:
            self.save_dead_item(each)

    # 得到某一帧的相对时间
    def get_time(self, frame_count: int):
        return frame_count / self.fps * 1000

    def update_flow(self, period_in_second, frame_count: int, maximum_allowance_per_minute=20):
        this_time = self.get_time(frame_count)
        # print(this_time // 1000, self.realtime // 1000)
        if this_time // 1000 - self.realtime // 1000 >= period_in_second:
            if len(self.flow_record) > 0:
                last_total_count = self.flow_record[-1][0]
                last_period_count = self.flow_record[-1][1]
                previous_total_count = self.item_count
                previous_period_count = self.item_count - last_total_count
                flow = previous_period_count * (60 // period_in_second)
                self.flow_record.append((previous_total_count, previous_period_count, flow, this_time))
            else:
                flow = self.item_count * (60 // period_in_second)
                self.flow_record.append((self.item_count, self.item_count, flow, this_time))
            self.overload = flow >= maximum_allowance_per_minute
            if self.overload:
                self.overload_record.append(self.flow_record[-1])
            self.realtime = this_time
        return self.overload


# -----------------------------------------
# 事件类，用来保存各种违规信息
class Event(object):

    class EventType(Enum):
        OVER_SPEED = 0
        INVERSE_DIRECTION = 1

    def __init__(self, event_id: int, item_id: int, event_time: int, event_frame, event_type):
        self.identification = event_id
        self.item = item_id
        self.time = event_time
        self.frame = event_frame
        self.type = event_type

    def save_event(self, video_name: str):
        file = open('%s\\event%04d.txt' % (video_name, self.identification), 'w')
        file.write("* 涉事id：%d\n" % self.item)
        file.write("* 涉事时间：%s\n" % Display.format_time(self.time))
        if self.type == Event.EventType.OVER_SPEED:
            ch_type = "超速"
        elif self.type == Event.EventType.INVERSE_DIRECTION:
            ch_type = "逆行"
        else:
            ch_type = "超速"
        file.write("* 肇事类型：%s\n" % ch_type)
        # cv.imwrite("%s\\event%04d" % (video_name, self.identification), self.frame)
        string = "%s\\event%04d.jpg" % (video_name, self.identification)
        cv.imencode('.jpg', self.frame)[1].tofile(string)


if __name__ == '__main__':
    camera = Camera(6, 30, 53, 30)
    print(camera.count_relative_height(540))
    print(camera.count_distance(2))
