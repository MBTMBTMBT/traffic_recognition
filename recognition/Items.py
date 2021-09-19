# coding=utf8
import cv2 as cv
import math
import os
from tools import Display, Geometry
from recognition import clpr_entry, clpr_recognition, type_classifier

classifier = type_classifier.Network((250 * 250, 128, 128, 128, 4))
abs_path = os.path.abspath('.')
# abs_path = abs_path.replace('\\', '\\\\')
classifier.load_parameters(abs_path + '\\mats\\softmax.npy')


class Item(object):

    # active_items = dataStructure.LinkedList()
    # inactive_items = dataStructure.LinkedList()

    # 定义每个被追踪物体的参数：ID、长度、宽度、平均速度、被跟丢次数等
    def __init__(self, identification: int, coord_x: int, coord_y: int,
                 width: int, height: int, start_time: int, cv_frame):
        self.identification = identification
        # self.location = Geometry.Point(coord_x, coord_y)
        self.rect = Geometry.Rect(coord_x, coord_y, width, height)
        self.break_count = 0
        self.tracker = cv.TrackerMOSSE_create()
        self.tracker.init(cv_frame, (coord_x, coord_y, width, height))
        self.trace = []
        self.real_trace = []
        self.move_length = []
        self.speed = []
        self.average_speed = 0
        self.is_overlapping = False
        self.remain = True
        self.lost_times = 0
        self.quick_shots = []
        self.take_quick_shot(cv_frame)
        self.start_time = start_time
        self.end_time = 0
        self.plates = {}
        self.predicted_plate = ""
        self.plate_count = 0
        self.plate_strs = []
        self.moving_forward = True
        self.forward_count = 0
        self.backward_count = 0
        self.in_event = False
        self.type = []
        self.die_without_recorded = False

    # def get_location(self) -> Geometry.Point:
    #     return self.location

    def get_type(self):
        records = [self.type.count(1), self.type.count(2), self.type.count(3)]  # self.type.count(0),
        return records.index(max(records)) + 1

    # 得到指定物体长、宽、大小
    def get_height(self) -> int:
        return self.rect.height

    def get_width(self) -> int:
        return self.rect.width

    def get_size(self) -> int:
        return self.rect.size()

    # 设定指定物体长度宽度
    def set_height(self, height: int):
        self.rect.height = height

    def set_width(self, width: int):
        self.rect.width = width

    def detect_type(self, pic):
        rect = self.rect
        x, y = rect.get_coord()
        w = rect.width
        h = rect.height
        roi = pic[y:y + h, x:x + w]
        rst = classifier.predict_pic(roi, (250, 250))
        # rst = rst.squeeze()
        self.type.append(rst)
        # print(rst)

    # 根据每一帧获得的图像，更新追踪器参数
    def update_tracker(self, frame, camera):
        success, box = self.tracker.update(frame)

        # 假如追踪器没有跟上移动物体，则lost_times + 1
        # 假如追踪器已经超过两次追踪到的移动物体，则可以通过物体的移动方向预测其下一帧的位置
        # 如果追踪器跟丢次数小于10，则可以通过预测移动物体位置的方式，让追踪器寻找移动物体
        # 假如跟丢次数过多，则舍弃当前跟踪器
        if not success:
            self.lost_times += 1
            if len(self.trace) >= 2:
                box = [0, 0, 0, 0]
                # print(self.trace[len(self.trace) - 1][0] * 2 - self.trace[len(self.trace) - 2][0])
                if self.lost_times <= 15:
                    box[0] = self.trace[len(self.trace) - 1][0] * 2 - self.trace[len(self.trace) - 2][0]
                    box[1] = self.trace[len(self.trace) - 1][1] * 2 - self.trace[len(self.trace) - 2][1]
                    box[2] = self.trace[len(self.trace) - 1][2]
                    box[3] = self.trace[len(self.trace) - 1][3]
            else:
                self.lost_times = 0

        # 得到每个追踪器的当前帧下的位置、长度、宽度、追踪路径长度
        self.rect.location = Geometry.Point(int(box[0]), int(box[1]))
        self.rect.width = int(box[2])
        self.rect.height = int(box[3])
        self.trace.append(box)
        rect = Geometry.Rect(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        distance = self.get_distance(camera, frame.shape[0])
        horizontal_offset \
            = self.get_horizontal_offset(camera, frame.shape[1], rect.get_mid_point().get_coord()[0])
        self.real_trace.append((distance, horizontal_offset))
        if len(self.real_trace) > 1:
            now = self.real_trace[len(self.real_trace) - 1]
            last = self.real_trace[len(self.real_trace) - 2]
            length = math.sqrt((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2)
            # print(now[0], last[0])
            self.move_length.append(length)

        # 运动方向检测
        if self.trace[-1][1] - self.trace[-len(self.trace) // 2][1] <= 0:  # -len(self.real_trace) // 2
            self.moving_forward = True
            self.backward_count = 0
            self.forward_count += 1
        else:
            self.moving_forward = False
            self.backward_count += 1
            self.forward_count = 0
        # print(self.forward_count, self.backward_count)
        return success, box

    # 通过移动物体的目前移动距离、当前时间及移动物体出现时间计算该物体当前平均速度
    def get_speed(self, video, frame_count):
        if len(self.move_length) > 2:
            distance = self.move_length[-2]
            speed = distance * video.fps
            self.speed.append(speed)
            now = self.real_trace[-1]
            last = self.real_trace[0]
            total_distance = math.sqrt((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2)
            '''
            if self.identification == 25:
                print(self.identification)
                print(now, last)
                print(total_distance, frame_count / video.fps)
                print("================")
                '''
            self.average_speed = total_distance / (frame_count / video.fps - self.start_time / 1000)
            # print(total_distance)
            return speed, self.average_speed
        else:
            return 0, 0

    # 判断所追踪的物体是否在移动
    def is_moving(self):
        try:
            x, y, _, _ = self.trace[len(self.trace) - 10]
            x1, y1, _, _ = self.trace[len(self.trace) - 1]
            return x1 != x or y1 != y
        except IndexError:
            return True

    # 给物体截图拍照
    def take_quick_shot(self, cv_frame):
        x = self.rect.get_coord()[0]
        y = self.rect.get_coord()[1]
        w = self.get_width()
        h = self.get_height()
        # print(x, y, self.get_width(), self.get_height())
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            cut = cv_frame[y: y + h, x: x + w]
            self.quick_shots.append(cut)
            return cut

    def take_quick_shot_with_highest_resolution(self, cv_frame, x, y, w, h):
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            cut = cv_frame[y: y + h, x: x + w]
            self.quick_shots.append(cut)
            return cut

    # 将拍摄得到的照片按照片大小排序
    def sort_quick_shots(self):
        self.quick_shots.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

    # 在画面中展示拍摄到的照片
    def display_quick_shots(self):
        count = 0
        for each in self.quick_shots:
            string = "%d %d" % (self.identification, count)
            cv.imshow(string, each)
            count += 1

    # 得到物体与摄像机的竖直高度距离
    def get_distance(self, camera, pixel_height=1080):
        relative_height = camera.count_relative_height(pixel_height - self.rect.get_coord_opposite()[1])
        return camera.count_distance(relative_height)

    # 得到物体与摄像机间的水平偏移量
    def get_horizontal_offset(self, camera, pixel_width=1920, pixel_height=1080):
        pixel = self.rect.get_mid_point().get_coord()[0]
        return camera.count_horizontal_offset(self.get_distance(camera, pixel_height), pixel, pixel_width)

    # 记录物体消除的时间
    def suicide(self, time_in_ms: int):
        self.end_time = time_in_ms

    # 记录车牌的识别结果，为反复训练所用
    def record_plate_recognition(self, image, target_video_path=None) -> (bool, str):
        success, rst, roi, cards = Item.predict_plate(image)
        # print(success, rst)
        if success:
            # self.plates.append(rst)
            if rst in self.plates.keys():
                self.plates[rst] += 1
            else:
                self.plates[rst] = 1
            if target_video_path is not None:
                try:
                    os.mkdir(target_video_path + "\\id%d-plate" % self.identification)
                except WindowsError:
                    pass
                # cv.imwrite(target_video_path + "\\id%d-plate\\plate-%d.png"
                #            % (self.identification, self.plate_count), roi)
                cv.imencode('.png', roi)[1].tofile(target_video_path + "\\id%d-plate\\plate-%d.png"
                                                   % (self.identification, self.plate_count))
                self.plate_strs.append(rst)

                rst_copy = rst[0:2] + rst[3:]
                for i in range(len(cards)):
                    if i == 0:
                        string = clpr_recognition.provinces[clpr_recognition.provinces.index(rst_copy[0]) - 1]
                    else:
                        string = rst_copy[i]
                    try:
                        # cv.imwrite(, cards[i])
                        cv.imencode('.png', cards[i])[1].tofile(target_video_path + "\\id%d-plate\\%s-%d.png"
                                                                % (self.identification, string, self.plate_count))
                    except:
                        # cv.imshow("unwriteable", cards[i])
                        # cv.waitKey(0)
                        continue
                self.plate_count += 1
        max_num = 0
        max_key = rst
        # print(self.plates)
        for key in self.plates.keys():
            if self.plates[key] > max_num:
                max_num = self.plates[key]
                max_key = key
        else:
            self.predicted_plate = max_key
        # print(self.predicted_plate)
        return success, rst

    def save_pic(self, num: int, pic):
        rect = self.rect
        x, y = rect.get_coord()
        w = rect.width
        h = rect.height
        roi = pic[y:y + h, x:x + w]
        cv.imwrite('masks\\%d.png' % num, roi)

    @staticmethod
    def static_detect_type(roi):
        rst = classifier.predict_pic(roi, (250, 250))
        return rst

    # 识别车牌 - 静态方法
    @staticmethod
    def predict_plate(image) -> (bool, str):
        try:
            rst, roi, cards = clpr_entry.clpr_main(image)
            # print(rst)
            # if rst == '':
            #     return False, rst, roi
            if not rst == '':
                return True, rst, roi, cards
            else:
                return False, rst, roi, cards
        except:
            return False, '', None, None

    # 避免由于两个物体重叠时调整追踪器矩形框大小而跟丢当前被追踪物体
    # 因此设置“are_overlapping”参数，若其为“True”状态，则该物体追踪器矩形框与其他物体追踪器矩形框有重叠；反之亦然
    @staticmethod
    def are_overlapping(item1, item2) -> bool:
        return Geometry.Rect.are_overlapping(item1.rect, item2.rect)

    # 重置每个物体重叠状态
    @staticmethod
    def set_all_not_overlapping(items: []):
        for each in items:
            each.is_overlapping = False

    # 匹配两个碰撞物体，若两个物体相互重叠，则都设成重叠状态
    @staticmethod
    def overlap_match(items: []):
        for a in items:
            for b in items:
                if a is b:
                    continue
                # elif a.is_overlapping and b.is_overlapping:
                #     continue
                else:
                    if Item.are_overlapping(a, b):
                        a.is_overlapping = True
                        b.is_overlapping = True

    def __str__(self):
        return "ID: %d, Item of rect with " % self.identification + str(self.rect)
