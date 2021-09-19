# -*- coding : utf-8 -*-
# coding: utf-8
import cv2
import threading
from recognition import Items, Video
from tools import Display, Geometry, Similarity, Get_line


# -----------------------------------------
class RecognitionThreadPool:
    mask_count = 0

    def __init__(self):
        self.pool = {}
        self.count = 0

    # 线程一加上就会开始分析，如果导入重复视频会直接报错
    # 但这个不能检查导入的到底是不是视频，所以可能需要前端查一下
    def add_video(self, video_input: str, camera_mod=0, boundary_mod=0,
                  dilate_iteration=4, dilate_ellipse=(3, 3), size_mod=0, flow_allowance=60, speed_limit=60,
                  stop_line_detect=False, flow_detect=True, direction_detect=False, speed_detect=True,
                  light=(10.0, 10.0, 10.0), direct_line_bottom=7 / 30, direct_line_top=11 / 30, front_end=None):
        if video_input in self.pool.keys():
            raise RuntimeError("%s正在被分析！" % video_input)
        else:
            self.pool[video_input] = RecognitionThread(video_input, self.count, camera_mod, boundary_mod,
                                                       dilate_iteration,
                                                       dilate_ellipse, size_mod, flow_allowance, speed_limit,
                                                       stop_line_detect, flow_detect, direction_detect, speed_detect,
                                                       light, direct_line_bottom, direct_line_top, front_end)
            self.pool[video_input].start()
            self.count += 1

    # 这个是专门查看是不是分析完成的，只返回结果，没有别的影响
    def has_finished(self, video_input: str):
        if self.pool[video_input] is not None:
            return self.pool[video_input].finished
        else:
            return True

    # 这个目前只返回实际帧数的比例和百分比
    def get_process(self, video_input: str):
        string, percentage = self.pool[video_input].get_progress_rate()
        return string, percentage

    # 这个是用于自查的（但也需要人为调用），会把跑完的线程清除掉节省空间
    def check_and_del(self):
        for key in self.pool.keys():
            if self.has_finished(key):
                del self.pool[key]
                print("%s分析已完成！" % key)

    # 返回现有视频的名称列表
    def get_names(self):
        return self.pool.keys()

    # 在导入阶段是否显示画面
    def show_frame(self, show: bool):
        for key in self.pool.keys():
            if self.pool[key] is not None:
                self.pool[key].show_frame = show

    # 全部线程关闭！配合前端使用
    def stop_all(self):
        for key in self.pool.keys():
            self.pool[key].stop = True


# -----------------------------------------
# 分析模式说明：
# 摄像头模式默认为0，另一模式为45度俯角，由于对摄像头实际参数不了解，两个值均为示意值
# 识别框模式：0 - 适合较近距离高俯角的情况，视野内只有行车道；1 - 适合视野较大、低俯角的情况，识别区域只包含路口中心；2 - 可能还有别的模式，预留空间
# 识别面积范围；0 - 适合距离近，物体面积变化不大的情况；1 - 适合距离远，或物体大小变化比较明显；2 - 预留
# 功能；停车线和红绿灯检测 - 默认关；流量监测 - 默认开；行车方向检测 - 默认关；速度检测 - 默认开
# -----------------------------------------
class RecognitionThread(threading.Thread):

    def __init__(self, video_input: str, thread_id: int, camera_mod=0, boundary_mod=0,
                 dilate_iteration=4, dilate_ellipse=(3, 3), size_mod=0, flow_allowance=60, speed_limit=40,
                 stop_line_detect=False, flow_detect=True, direction_detect=False, speed_detect=True,
                 light=(10.0, 10.0, 10.0), direct_line_bottom=7 / 30, direct_line_top=11 / 30, front_end=None):
        # light的第一个值是第一次红灯的时间（视频开始后第多少秒），第二个值是红灯时长，第三个是绿灯时长
        super().__init__()
        self.thread_id = thread_id
        self.stop = False  # 用于控制全局线程统一退出
        self.video_input = video_input
        self.present_frame_num = 1
        self.total_frame_num = 0
        self.finished = False
        self.show_frame = True  # 前端会引导对这个值进行修改来达到显示或不显示opencv界面的作用
        self.front_end = front_end

        self.SPEED_DETECT = speed_detect
        self.FLOW_DETECT = flow_detect
        self.STOP_LINE_DETECT = stop_line_detect
        self.DIRECTION_DETECT = direction_detect

        # 摄像机的高度，视角和俯角 - 参考值
        if camera_mod == 0:
            self.CAMERA_HEIGHT = 6
            self.CAMERA_VERTICAL_ANGLE = 20
            self.CAMERA_HORIZONTAL_ANGLE = 38
            self.CAMERA_DEPRESSION_ANGLE = 30
        elif camera_mod == 1:
            self.CAMERA_HEIGHT = 6
            self.CAMERA_VERTICAL_ANGLE = 20
            self.CAMERA_HORIZONTAL_ANGLE = 38
            self.CAMERA_DEPRESSION_ANGLE = 45
        else:
            self.CAMERA_HEIGHT = 6
            self.CAMERA_VERTICAL_ANGLE = 20
            self.CAMERA_HORIZONTAL_ANGLE = 38
            self.CAMERA_DEPRESSION_ANGLE = 30

        # 识别区的左上坐标和宽、高 - 比例值，不是真实值
        if boundary_mod == 0:
            self.BOUNDARY_X = 12 / 1
            self.BOUNDARY_Y = 12 / 1
            self.BOUNDARY_WIDTH = 12 / 10
            self.BOUNDARY_HEIGHT = 12 / 10
        elif boundary_mod == 1:
            self.BOUNDARY_X = 12 / 2
            self.BOUNDARY_Y = 12 / 3
            self.BOUNDARY_WIDTH = 12 / 8
            self.BOUNDARY_HEIGHT = 12 / 6
        else:
            self.BOUNDARY_X = 12 / 0.0001
            self.BOUNDARY_Y = 12 / 2
            self.BOUNDARY_WIDTH = 12 / 12
            self.BOUNDARY_HEIGHT = 12 / 6

        # 进行车牌识别的识别线
        # 即 - 在此线下方的截图才会进行车牌识别
        # 注意这是一个关于屏幕高度的比例值
        self.PLATE_RECOGNITION_LINE = 0.2

        # 二值图扩张的迭代次数和椭圆大小
        self.DILATE_ITERATION = dilate_iteration
        self.DILATE_ELLIPSE = dilate_ellipse

        # 判定为实际物体的大小限制
        if size_mod == 0:
            # 下线是像素点面积
            self.ITEM_CONFIRM_SIZE_LOWER_LIMIT = 8000
            # 注意 - 上限是占全屏面积的比例
            self.ITEM_CONFIRM_SIZE_UPPER_LIMIT = 1 / 7
        elif size_mod == 1:
            self.ITEM_CONFIRM_SIZE_LOWER_LIMIT = 6000
            self.ITEM_CONFIRM_SIZE_UPPER_LIMIT = 1 / 8
        else:
            self.ITEM_CONFIRM_SIZE_LOWER_LIMIT = 8000
            self.ITEM_CONFIRM_SIZE_UPPER_LIMIT = 1 / 7

        # 最小判定帧数 - 即一个物体存在的帧数超过此值才被判定为真正存在
        self.SMALLEST_FRAME_NUMBER_LIMIT = 10

        # 每多少帧判定一次物体在不在运动
        self.MOVING_OBJECT_JUDGE_FREQUENCY = 10

        self.ACCEPTABLE_OVERLAPPING_RATIO = 0.9

        # 用于对不处于“保护态”的物体进行大小更新，以及对相邻区域内的新发现物体做判定
        # “保护态” - 是针对处于重叠状态的物体进行的保护设定
        #
        # 对识别到的物体尺寸变化做判定使用 - 超出此范围的比例会被认为是误判
        self.RECOGNITION_HEIGHT_WIDTH_RATE_UPPER_LIMIT = 2.5
        self.RECOGNITION_HEIGHT_WIDTH_RATE_LOWER_LIMIT = 0.4
        #
        # 计算的相似度超过此值的物体会被认为是同一物体
        self.SAME_ITEM_RECOGNITION_SIMILARITY = 0.5
        #
        # 尺寸更新时的宽高变化接受程度
        self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE = 0.5
        #
        # 小于此值的物体会被当成新物体进行跟踪
        # 注意 - 与上面的相似度是不同的值
        self.DIFFERENT_ITEM_RECOGNITION_SIMILARITY = 0.2
        #
        # 尺寸变化的接受程度
        # 如 - 0.3代表变化后的面积占变化前的0.3-3.33之间被认为有效
        self.SIZE_UPDATE_RATE_ALLOWANCE = 0.3

        # 保留的截图数量
        self.QUICK_SHOT_KEEP_NUM = 5

        # 截图频率 - 每多少帧截一次
        self.QUICK_SHOT_TAKEN_FREQUENCY = 5

        # 车流估算的频率和报警峰值
        self.FLOW_UPDATE_PERIOD = 5
        self.MAXIMUM_FLOW_ALLOWANCE = flow_allowance  # 次每分钟

        # 限速：公里每小时
        self.SPEED_LIMIT = speed_limit

        self.STOP_LINE_LEFT = 1 / 2
        self.STOP_LINE_RIGHT = 1 / 2

        self.DIRECTION_LINE_BOTTOM = direct_line_bottom
        self.DIRECTION_LINE_TOP = direct_line_top

    def get_progress_rate(self):
        string, percentage = Display.get_progress_rate(self.present_frame_num, self.total_frame_num)
        return string, percentage

    def run(self):
        # 初始化
        # -----------------------------------------
        camera = Video.Camera(self.CAMERA_HEIGHT, self.CAMERA_VERTICAL_ANGLE, self.CAMERA_HORIZONTAL_ANGLE,
                              self.CAMERA_DEPRESSION_ANGLE)
        background_subtractor = cv2.createBackgroundSubtractorKNN()
        video = Video.Video(cv2.VideoCapture(self.video_input), video_name=self.video_input)
        self.total_frame_num = video.total_frames_num
        shape = (video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # 1k下的shape
        shape_original = shape
        shape = (1080, 1920)

        rect_y = shape[0] // self.BOUNDARY_Y
        rect_x = shape[1] // self.BOUNDARY_X
        boundary = Geometry.Rect(rect_x, rect_y, shape[1] // self.BOUNDARY_WIDTH, shape[0] // self.BOUNDARY_HEIGHT)
        item_count = 0
        event_count = 0
        plate_recognition_thread_pool = {}
        save_died_item_th = None
        count_th = 0

        # 做方向判定的分界线的两点
        # 这里把这两个点移到上面来了，不用每次都算一遍
        # 这里不能再用video那个类了，不然分辨率会出问题
        bottom = (int(shape[1] * self.DIRECTION_LINE_BOTTOM), int(shape[0]))
        top = (int(shape[1] * self.DIRECTION_LINE_TOP), 0)
        while True:
            try:
                if self.DIRECTION_DETECT:
                    _, first_frame = video.video_capture.read()
                    first_frame = cv2.resize(first_frame, (1280, 720))
                    x0, y0, x1, y1 \
                        = Get_line.draw_line(first_frame, winname("id: %d %s_frame" % (self.thread_id, video.video_name)))
                    print(x0, y0, x1, y1)
                    x0, y0 = Video.Frame.position_pixel_transfer((x0, y0), (1280, 720), (1920, 1080))
                    x1, y1 = Video.Frame.position_pixel_transfer((x1, y1), (1280, 720), (1920, 1080))
                    print(x0, y0, x1, y1)
                    if y0 >= y1:
                        top = (x0, y0)
                        bottom = (x1, y1)
                    else:
                        top = (x1, y1)
                        bottom = (x0, y0)
                    if top[0] == bottom[0]:
                        top = (top[0], 0)
                        bottom = (bottom[0], 1080)
                    else:
                        k = (top[1] - bottom[1]) / (top[0] - bottom[0])
                        b = (top[1] + bottom[1] - k * (top[0] + bottom[0])) / 2
                        top = (int(- b / k), 0)
                        bottom = (int((1080 - b) / k), 1080)
                    break
            except Exception as e:
                print(e)
                continue
        # -----------------------------------------

        for frame_count in range(video.total_frames_num):
            try:
                # 通过背景相减法得到正在移动的物体
                self.present_frame_num = frame_count + 1
                success, frame = video.video_capture.read()
                if not success:
                    break

                try:
                    if frame_count % 3 == 0:
                        video.add_frame_to_video(copy)
                        continue
                except UnboundLocalError as e:
                    print(e)

                # 这里也做了低分辨率的替换
                frame_original = frame
                frame = cv2.resize(frame, (1920, 1080))

                frame = Video.Frame(frame, frame_count + 1, video)
                copy = frame.cv_frame.copy()
                # copy_720 = cv2.resize(copy, (720, 1280))

                # 高斯模糊图像，避免摄像噪点产生的移动判断误差
                frame_blur = cv2.GaussianBlur(frame.cv_frame.copy(), (13, 13), 0)  # 高斯模糊

                # 将每一帧与上一帧做减法，识别出移动部分与背景部分
                #    - Opencv的KNN法据说采用了一些机器学习的算法，生成的图像噪点比较低
                #
                # 注意！一定要说明的是，KNN会把长时间静止或视频一开始处于静止的物体判定为背景的一部分，
                # 效果有一点像电子蛙眼 - 对静止事物不敏感。因此一个物体长时间停留（比如等红灯），就会无法被继续追踪
                # 解决的最好办法就是通过拍摄一张空的道路图片作为被减数，而不是通过KNN这种实时判断的工具
                # 但其实也还会有其他问题，例如对摄像头的抖动十分敏感等等，但如果后面有机会，还是倾向于使用全空的街道背景作为底图的方法
                mask = background_subtractor.apply(frame_blur)  # 由KNN产生

                # 将模糊的灰色部分(不能明确是否为移动物体)二值化为白色(移动部分)或黑色(背景部分)
                th = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)[1]  # 二值化

                # 前面这里新加一个腐蚀
                erode = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                  iterations=2)

                # 将白色部分(明确的移动部分)进行扩张，把一些破碎的图像进行联通，避免由于二值化导致的移动边缘识别误差
                # 但这个值的取值实际上比较矛盾，扩张的这两个参数值大了又会造成误连，小了很多地方连不到一起，是破碎的，选择上会比较困难
                dilated = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.DILATE_ELLIPSE),
                                     iterations=self.DILATE_ITERATION)  # 扩张
                # cv2.imshow("dilated", dilated)

                # 框选出二值图白色部分的轮廓 - 即判定为移动物体的部分
                contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 避免两个移动物体过近或重叠，若跟踪器重叠，则两个跟踪器大小锁死，单独大小更新判断
                Items.Item.overlap_match(video.items)

                # -----------------------------------------
                # 这个地方不符合要求的轮廓会被反复判断多次，有待优化
                # 这个地方是为了给保护态的物体优先更新大小
                # 原理为如果追踪框内仅有一个满足要求的二值图轮廓，就直接把保护态的跟踪框更新为该轮廓框出的矩形
                contours_copy = contours[:]
                for each in video.items:
                    try:
                        # 对于每一个处于重叠状态的物体
                        if each.is_overlapping:
                            inside_list = []
                            for c in contours_copy:
                                # 筛选移动物体大小（若过小，则判定为噪点，若过大，则判定为画面抖动造成的背景移动）
                                if not shape[0] * shape[1] * self.ITEM_CONFIRM_SIZE_UPPER_LIMIT > cv2.contourArea(c) \
                                       > self.ITEM_CONFIRM_SIZE_LOWER_LIMIT:
                                    continue
                                # 对在判定区中的移动物体框上矩形
                                (x, y, w, h) = cv2.boundingRect(c)
                                # 判断物体的长宽比是否正常 - 可以排除一些框的是马路牙子或者栏杆等等的东西
                                if not self.RECOGNITION_HEIGHT_WIDTH_RATE_UPPER_LIMIT > h / w \
                                       > self.RECOGNITION_HEIGHT_WIDTH_RATE_LOWER_LIMIT:
                                    continue
                                # 根据移动物体是否在判定区中进行筛选
                                if not Geometry.Rect.has_inside(boundary, Geometry.Rect(x, y, w, h).get_mid_point()):
                                    continue
                                contour_rect = Geometry.Rect(x, y, w, h)

                                # 如果轮廓在跟踪框内，就记录下来
                                if each.rect.has_rect_inside(contour_rect):
                                    inside_list.append(contour_rect)

                            # 如果一个轮廓都不包括，很可能是跟丢了，在这里就不再处理了
                            if len(inside_list) == 0:
                                continue

                            # 如果只包括这一个轮廓，很可能就是他自己
                            elif len(inside_list) == 1:
                                coord = each.rect.get_coord()
                                height = each.rect.height
                                width = each.rect.width
                                item_roi = frame_blur[int(coord[1]):int(coord[1] + height),
                                           int(coord[0]):int(coord[0] + width)]
                                rect = inside_list[0]
                                x, y = rect.get_coord()
                                w = rect.width
                                h = rect.height
                                contour_roi = frame_blur[int(y):int(y + h), int(x):int(x + w)]
                                similarity = Similarity.classify_hist_with_split(item_roi, contour_roi)

                                # 满足相似度和其它条件的话直接就更新到这个轮廓上了
                                if similarity[0] >= self.SAME_ITEM_RECOGNITION_SIMILARITY \
                                        and self.SIZE_UPDATE_RATE_ALLOWANCE \
                                        < each.rect.size() / (w * h) < 1 / self.SIZE_UPDATE_RATE_ALLOWANCE \
                                        and self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE \
                                        < (each.rect.width / each.rect.height) / (w / h) \
                                        < 1 / self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE:
                                    each.tracker = cv2.TrackerMOSSE_create()
                                    each.tracker.init(frame.cv_frame, (x, y, w, h))
                                    each.remain = True

                            # 如果包括不止这一个轮廓，那就在这些轮廓里找到相似度最高的
                            else:
                                coord = each.rect.get_coord()
                                height = each.rect.height
                                width = each.rect.width
                                item_roi = frame_blur[int(coord[1]):int(coord[1] + height),
                                           int(coord[0]):int(coord[0] + width)]
                                highest_similarity = 0
                                highest_one = None

                                # 对于每个轮廓，识别它的相似度
                                for each_rect in inside_list:
                                    rect = each_rect
                                    x, y = rect.get_coord()
                                    w = rect.width
                                    h = rect.height
                                    contour_roi = frame_blur[int(y):int(y + h), int(x):int(x + w)]
                                    similarity = Similarity.classify_hist_with_split(item_roi, contour_roi)
                                    similarity = similarity[0]
                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        highest_one = each_rect
                                x, y = highest_one.get_coord()
                                w = highest_one.width
                                h = highest_one.height

                                # 不过即使找到了，还是要判断一下满不满足条件，满足条件才进行保留
                                if highest_similarity > self.SAME_ITEM_RECOGNITION_SIMILARITY * 0.8 \
                                        and self.SIZE_UPDATE_RATE_ALLOWANCE \
                                        < each.rect.size() / (w * h) < 1 / self.SIZE_UPDATE_RATE_ALLOWANCE \
                                        and self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE \
                                        < (each.rect.width / each.rect.height) / (w / h) \
                                        < 1 / self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE:
                                    # 判断一下类型
                                    each.tracker = cv2.TrackerMOSSE_create()
                                    each.tracker.init(frame.cv_frame, (x, y, w, h))
                                    each.remain = True
                    except Exception as e:
                        print(e)
                        continue

                # -----------------------------------------
                # 后面这部分就是对那些没有锁死的物体进行判断了
                for c in contours:
                    try:
                        # 筛选移动物体大小（若过小，则判定为噪点，若过大，则判定为画面抖动造成的背景移动）
                        if shape[0] * shape[1] * self.ITEM_CONFIRM_SIZE_UPPER_LIMIT > cv2.contourArea(
                                c) > self.ITEM_CONFIRM_SIZE_LOWER_LIMIT:
                            # 对在判定区中的移动物体框上矩形
                            (x, y, w, h) = cv2.boundingRect(c)
                            # 判断物体的长宽比是否正常 - 可以排除一些框的是马路牙子或者栏杆等等的东西
                            if not self.RECOGNITION_HEIGHT_WIDTH_RATE_UPPER_LIMIT > h / w \
                                   > self.RECOGNITION_HEIGHT_WIDTH_RATE_LOWER_LIMIT:
                                continue
                            # 根据移动物体是否在判定区中进行筛选
                            if Geometry.Rect.has_inside(boundary, Geometry.Rect(x, y, w, h).get_mid_point()):
                                # 匹配每个移动物体和它更新后的追踪矩形
                                for each in video.items:
                                    coord = each.rect.get_coord()
                                    width = each.rect.width
                                    height = each.rect.height
                                    each_rect = Geometry.Rect(coord[0], coord[1], width, height)
                                    box_rect = Geometry.Rect(x, y, w, h)
                                    # oa, ob = Geometry.Rect.overlapping_ratio(each_rect, box_rect)
                                    # overlapping_ratio = min(oa, ob)

                                    # 下面开始尝试更新跟踪框的大小，以及开启一些新的跟踪
                                    if Geometry.Rect.are_overlapping(each_rect, box_rect):
                                        # 忽略已被锁死的移动物体
                                        if each.is_overlapping:
                                            break
                                        else:
                                            each_roi = frame_blur[int(coord[1]):int(coord[1] + height),
                                                       int(coord[0]):int(coord[0] + width)]
                                            box_roi = frame_blur[int(y):int(y + h), int(x):int(x + w)]
                                            similarity = Similarity.classify_hist_with_split(each_roi, box_roi)
                                            # 判断两个画面相似度，返回-1说明某一个画面有问题，则直接略过
                                            if similarity == -1:
                                                break
                                            # 相似度满足的情况下，若追踪框大小变化在可接受的范围内，我们认为新确定的轮廓为物体在新一帧内的更新轮廓
                                            # 对跟踪器进行重置，并保留此物体
                                            if similarity[0] >= self.SAME_ITEM_RECOGNITION_SIMILARITY \
                                                    and self.SIZE_UPDATE_RATE_ALLOWANCE \
                                                    < each.rect.size() / (w * h) < 1 / self.SIZE_UPDATE_RATE_ALLOWANCE \
                                                    and self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE \
                                                    < (each.rect.width / each.rect.height) / (w / h) \
                                                    < 1 / self.ITEM_TRACE_HEIGHT_WIDTH_CHANGE_RATE:
                                                each.tracker = cv2.TrackerMOSSE_create()
                                                each.tracker.init(frame.cv_frame, (x, y, w, h))
                                                each.remain = True
                                            # 若相似度太低，我们认为这个物体是新出现在这个画面中的物体，我们将它记为新的移动物体
                                            elif similarity[0] < self.DIFFERENT_ITEM_RECOGNITION_SIMILARITY: \
                                                    # and overlapping_ratio < self.ACCEPTABLE_OVERLAPPING_RATIO:
                                                temp_x, temp_y = Video.Frame.position_pixel_transfer((x, y), (1920, 1080),
                                                                                (shape_original[1], shape_original[0]))
                                                temp_w = int(w * shape_original[1] // 1920)
                                                temp_h = int(h * shape_original[0] // 1080)
                                                original_pixel = ()  # x, y, w, h
                                                # temp_roi = frame.cv_frame[y: y + h, x: x + w]
                                                temp_roi = frame_original[temp_y: temp_y + temp_h, temp_x: temp_x + temp_w]
                                                rst = Items.Item.static_detect_type(temp_roi)
                                                # cv2.imwrite('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\masks\\'
                                                #             + str(int(t.time() * 1000)) + '.png', temp_roi)
                                                if rst != 0:
                                                    # if True:
                                                    time = video.get_time(frame_count)
                                                    item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                                                    item.type.append(rst)  # 很重要，去掉了会出大问题
                                                    item.remain = True
                                                    item.is_overlapping = True
                                                    video.items.append(item)
                                                    if rst == 1:
                                                        item_count += 1
                                                    cv2.rectangle(copy, (x, y), (x + w, y + h), (128, 128, 128), 2)
                                            break
                                # 若跟踪框中的物体不与上一帧被追踪到的移动物体重叠，则我们认为这个物体是新出现在画面中的物体，我们将它记为新的移动物体
                                else:
                                    temp_x, temp_y = Video.Frame.position_pixel_transfer((x, y), (1920, 1080),
                                                                                         (shape_original[1],
                                                                                          shape_original[0]))
                                    temp_w = int(w * shape_original[1] // 1920)
                                    temp_h = int(h * shape_original[0] // 1080)
                                    original_pixel = ()  # x, y, w, h
                                    # temp_roi = frame.cv_frame[y: y + h, x: x + w]
                                    temp_roi = frame_original[temp_y: temp_y + temp_h, temp_x: temp_x + temp_w]
                                    rst = Items.Item.static_detect_type(temp_roi)
                                    # cv2.imwrite('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\masks\\'
                                    #             + str(int(t.time() * 1000)) + '.png', temp_roi)
                                    if rst != 0:
                                        # if True:
                                        time = video.get_time(frame_count)
                                        item = Items.Item(item_count, x, y, w, h, time, frame.cv_frame)
                                        item.type.append(rst)  # 很重要，去掉了会出大问题
                                        item.remain = True
                                        video.items.append(item)
                                        if rst == 1:
                                            item_count += 1
                                        cv2.rectangle(copy, (x, y), (x + w, y + h), (128, 128, 128), 2)
                    except Exception as e:
                        print(e)
                        continue

                # 对于每个物体执行一遍取消锁死
                Items.Item.set_all_not_overlapping(video.items)

                # 判断每个移动物体是否在判断区内
                # 若物体在判定区内，我们对这个物体每隔一定帧数帧判断一次它是否在运动中
                # 若物体不动，我们认为它是被误判的背景部分或者追踪器跟丢物体的情况，在移动物体列表中删除此物体
                # 注：此方法将把等红绿灯的移动车辆判定为不动的物体，从而删去本该被追踪的移动物体
                # 这是一个排除和清楚误判物体的有效方法，但存在上述的缺陷，如果有时间我们还是希望能够再做改进
                for each in video.items:
                    success, box = each.update_tracker(frame.cv_frame, camera)
                    if Geometry.Rect.has_inside(boundary,
                                                Geometry.Rect(box[0], box[1], box[2], box[3]).get_mid_point()):
                        if frame.serial_num % self.MOVING_OBJECT_JUDGE_FREQUENCY != 0:
                            each.remain = True
                        elif each.is_moving():
                            each.remain = True
                        else:
                            each.remain = False
                    else:
                        each.remain = False

                    if frame_count % 2 == 0 or frame_count % 5 == 0:
                        detect_type_th = threading.Thread(target=each.detect_type(frame.cv_frame))
                        detect_type_th.start()
                        # print(each.type)
                        # each.detect_type(frame.cv_frame)

                    if len(each.type) == 1:
                        if each.type[0] == 0:
                            each.remain = False
                            each.die_without_recorded = True
                    elif len(each.type) >= 1:
                        if each.type.count(0) / len(each.type) >= 0.8:
                            each.remain = False
                            # To do: True
                            each.die_without_recorded = True
                        elif each.type.count(0) / len(each.type) >= 0.55:
                            each.remain = False
                            each.die_without_recorded = False

                # 把继续存在和不该存在的物体进行分类并放入相应的列表中等待后续处理
                new_items = []
                for each in video.items:

                    if each.remain:
                        new_items.append(each)

                        # 如果处于截图的时间，就截图 - 更理想的情况应该是对于每个物体依据其各自已经存在的帧数进行截图和识别
                        if len(each.trace) % self.QUICK_SHOT_TAKEN_FREQUENCY == 0:
                            x, y = each.rect.get_coord()
                            w = each.rect.width
                            h = each.rect.height
                            tmp_x, tmp_y = Video.Frame.position_pixel_transfer((x, y), (1920, 1080), ((shape_original[1],
                                                                                          shape_original[0])))
                            tmp_w = int(w * shape_original[1] // 1920)
                            tmp_h = int(h * shape_original[0] // 1080)
                            shot = each.take_quick_shot_with_highest_resolution(frame_original, tmp_x, tmp_y, tmp_w, tmp_h)
                            # shot = each.take_quick_shot(frame.cv_frame)
                            '''
                            try:
                                each.save_pic(RecognitionThreadPool.mask_count, frame.cv_frame)
                                RecognitionThreadPool.mask_count += 1
                            except cv2.error as e:
                                print(e)
                                pass
                            '''

                            # 并对中心点处于识别线一下的截图进行车牌识别
                            if each.rect.get_mid_point().get_coord()[1] \
                                    >= video.picture_rect.height * self.PLATE_RECOGNITION_LINE:

                                # 每一次识别新起一个线程
                                id_num = each.identification
                                if id_num in plate_recognition_thread_pool.keys():
                                    plate_recognition_thread_pool[id_num].join()
                                plate_recognition_thread_pool[id_num] \
                                    = threading.Thread(target=each.record_plate_recognition,
                                                       args=(shot, video.video_name))
                                plate_recognition_thread_pool[id_num].start()
                                # each.record_plate_recognition(shot, video.video_name)
                                # plate_success, plate_str = Items.Item.predict_plate(shot)
                                # print(plate_success, plate_str)
                            each.sort_quick_shots()

                            # 对截图从大到小进行排序，并保留前一定数量的截图
                            if len(each.quick_shots) >= self.QUICK_SHOT_KEEP_NUM:
                                pics = []
                                for j in range(self.QUICK_SHOT_KEEP_NUM):
                                    pics.append(each.quick_shots[j])
                                each.quick_shots = pics

                    # 物体被移除时，统计移除的时间并额外截一张图
                    elif len(each.trace) >= self.SMALLEST_FRAME_NUMBER_LIMIT:
                        id_num = each.identification
                        # 对于识别车牌的线程，也从线程池中移除
                        if id_num in plate_recognition_thread_pool.keys():
                            plate_recognition_thread_pool[id_num].join()
                            del plate_recognition_thread_pool[id_num]
                        time = video.get_time(frame_count)
                        each.suicide(time)
                        if not each.die_without_recorded:
                            video.died_items.append(each)
                        x, y = each.rect.get_coord()
                        w = each.rect.width
                        h = each.rect.height
                        tmp_x, tmp_y = Video.Frame.position_pixel_transfer((x, y), (1920, 1080), ((shape_original[1],
                                                                                                   shape_original[0])))
                        tmp_w = int(w * shape_original[1] // 1920)
                        tmp_h = int(h * shape_original[0] // 1080)
                        each.take_quick_shot_with_highest_resolution(frame_original, tmp_x, tmp_y, tmp_w, tmp_h)
                        # each.take_quick_shot(frame.cv_frame)
                        # each.display_quick_shots()
                video.items = new_items

                # 另起一个线程来保存被移除的物体的数据
                died_list = video.died_items
                if save_died_item_th is not None:
                    save_died_item_th.join()
                save_died_item_th = threading.Thread(target=video.save_dead_items(died_list))
                save_died_item_th.start()
                for each in video.died_items:
                    video.grave.append(each)
                video.died_items = []

                # 查看交通是否过载
                overload = video.update_flow(self.FLOW_UPDATE_PERIOD, frame_count, self.MAXIMUM_FLOW_ALLOWANCE)

                # 在画面中标记每个追踪器的ID，移动距离，水平距离及平均速度和车牌号
                for each in video.items:
                    each_type = each.get_type()
                    if each_type == 1:
                        _, average_speed = each.get_speed(video, frame_count + 1)
                        average_speed *= 3.6
                        rect_coord_x, rect_coord_y = each.rect.get_coord()
                        if average_speed <= self.SPEED_LIMIT or not self.SPEED_DETECT:
                            color = (255, 0, 0)
                        else:
                            color = (0, 0, 255)
                            copy = Display.put_chinese_string(copy, "超速！", (rect_coord_x + 5, rect_coord_y + 30), color)
                            if not each.in_event:
                                # print("event!")
                                event = Video.Event(event_count, each.identification, video.get_time(frame_count),
                                                    copy, Video.Event.EventType.OVER_SPEED)
                                event.save_event(video.video_name)
                                each.in_event = True
                                event_count += 1
                        cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), color, 2)
                        string_id = "ID: " + str(each.identification)
                        string_position = "Distance: %.2fm" % each.get_distance(camera, video.picture_rect.height) \
                                          + " horizontal: %.2fm" % each.get_horizontal_offset(camera)
                        string_speed = "Average speed: %.2fkm/h" % average_speed
                        if each.predicted_plate != '':
                            string_plate = "Plate: " + each.predicted_plate
                        else:
                            string_plate = "Plate: "
                        cv2.putText(copy, string_id, (rect_coord_x, rect_coord_y - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                        cv2.putText(copy, string_position, (rect_coord_x, rect_coord_y - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                        cv2.putText(copy, string_speed, (rect_coord_x, rect_coord_y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                        # cv2.putText(copy, string_plate, (rect_coord_x, rect_coord_y - 0),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                        copy = Display.put_chinese_string(copy, string_plate, (rect_coord_x, rect_coord_y - 28), color)
                        copy = Display.put_chinese_string(copy, "汽车", (rect_coord_x + 5, rect_coord_y + 5), color)
                    elif each_type == 2:
                        # 土耳其蓝
                        cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (140, 199, 0), 2)
                        copy = Display.put_chinese_string(copy, "摩托/自行车", (each.rect.get_coord()[0] + 5,
                                                                           each.rect.get_coord()[1] + 5), (140, 199, 0))
                    elif each_type == 3:
                        # print(0)
                        cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (255, 0, 255), 2)
                        # print(1)
                        copy = Display.put_chinese_string(copy, "行人", (each.rect.get_coord()[0] + 5,
                                                                           each.rect.get_coord()[1] + 5), (255, 0, 255))

                # 标记此视频的识别区，过载时做出报警
                # print(video.flow_record[-1][2])
                if self.FLOW_DETECT:
                    rect_coord_x, rect_coord_y = boundary.get_coord()
                    # print(video.flow_record[-1][2])
                    if overload:
                        cv2.rectangle(copy, boundary.get_coord(), boundary.get_coord_opposite(), (64, 128, 255), 4)
                        try:
                            string = "车流过载！约%d辆/分钟！" % video.flow_record[-1][2]
                            copy = Display.put_chinese_string(copy, string, (rect_coord_x, rect_coord_y - 38),
                                                              (64, 128, 255))
                        except Exception as e:
                            print(e)
                    else:
                        cv2.rectangle(copy, boundary.get_coord(), boundary.get_coord_opposite(), (255, 255, 0), 2)
                        if len(video.flow_record) > 0:
                            try:
                                string = "车流量：约%d辆/分钟" % video.flow_record[-1][2]
                            except Exception as e:
                                print(e)
                                string = "车流量：约0辆/分钟"
                        else:
                            string = "车流量：约0辆/分钟"
                        copy = Display.put_chinese_string(copy, string, (rect_coord_x, rect_coord_y - 38),
                                                          (255, 255, 0))
                    # rect_coord_x, rect_coord_y - 38
                else:
                    cv2.rectangle(copy, boundary.get_coord(), boundary.get_coord_opposite(), (255, 255, 0), 2)

                # 方向检测，主要就是为了检测逆行用的
                if self.DIRECTION_DETECT:
                    # bottom = (int(video.picture_rect.width * self.DIRECTION_LINE_BOTTOM),
                    #           int(video.picture_rect.height))
                    # top = (int(video.picture_rect.width * self.DIRECTION_LINE_TOP), 0)

                    cv2.line(copy, bottom, top, (255, 255, 0), 2)
                    line = Geometry.Line(bottom, top)
                    for each in video.items:
                        if each.get_type() != 1:
                            continue
                        mid = each.rect.get_mid_point()
                        if mid is None:
                            continue
                        if line.point_at_left(mid.get_coord()):
                            if each.forward_count >= 10:
                                rect_coord_x, rect_coord_y = each.rect.get_coord()
                                copy = Display.put_chinese_string(copy, "逆行！", (rect_coord_x + 65, rect_coord_y + 30),
                                                                  (0, 0, 255))
                                cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (0, 0, 255),
                                              2)
                                if not each.in_event:
                                    # print("event!")
                                    event = Video.Event(event_count, each.identification, video.get_time(frame_count),
                                                        copy, Video.Event.EventType.INVERSE_DIRECTION)
                                    event.save_event(video.video_name)
                                    each.in_event = True
                                    event_count += 1
                                pass
                            # print("left")
                        else:
                            if each.backward_count >= 10:
                                rect_coord_x, rect_coord_y = each.rect.get_coord()
                                copy = Display.put_chinese_string(copy, "逆行！", (rect_coord_x + 65, rect_coord_y + 5),
                                                                  (0, 0, 255))
                                cv2.rectangle(copy, each.rect.get_coord(), each.rect.get_coord_opposite(), (0, 0, 255),
                                              2)
                                if not each.in_event:
                                    print("event!")
                                    event = Video.Event(event_count, each.identification, video.get_time(frame_count),
                                                        copy, Video.Event.EventType.INVERSE_DIRECTION)
                                    event.save_event(video.video_name)
                                    each.in_event = True
                                    event_count += 1
                                pass
                            # print("right")

                # show frame
                # 把每一帧添加到生成的结果视频文件中
                video.add_frame_to_video(copy)
                if self.show_frame:
                    copy = cv2.resize(copy, (960, 540))
                    # dilated_copy = cv2.resize(dilated, (1280, 720))
                    # cv2.imshow(winname('%s_frame' % video.video_name), copy)
                    # cv2.imshow("id: %d mask" % self.thread_id, dilated_copy)
                    cv2.imshow(winname("id: %d %s_frame" % (self.thread_id, video.video_name)), copy)
                else:
                    cv2.destroyAllWindows()
                # cv2.imshow('blur', frame_blur)
                # cv2.imshow('mask', dilated)

                # quit on ESC button
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    cv2.destroyAllWindows()
                    self.show_frame = False
                    self.front_end.show_frame_bool = False
                    continue

                # 如果前端给出关闭信号，这个线程将立即跳出循环并开始保存
                if self.stop:
                    break

            except Exception as e:
                print(e)
                pass

        video.save_video_info()
        # 完成后标记此线程运行完成
        self.finished = True
        cv2.destroyWindow(winname("id: %d %s_frame" % (self.thread_id, video.video_name)))
        # cv2.destroyWindow("id: %d mask" % self.thread_id)
        video.video_write.release()
        if self.front_end is not None:
            self.front_end.update_list = True


def winname(name):
    return name.encode('gbk').decode(errors='ignore')


if __name__ == '__main__':
    thread_pool = RecognitionThreadPool()
    thread_pool.add_video('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_New\\video-3.mp4', direction_detect=True)
    # thread_pool.add_video('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_New\\video-02.mp4', boundary_mod=1, size_mod=1)
    # thread_pool.add_video('00014.mp4')
    pass
    # recognize('video-02.mp4')
    # recognize('MAH00057.mp4')
    # recognize('video-01.avi')
    # recognize('video-03.avi')
    # recognize('video-04.MP4')
