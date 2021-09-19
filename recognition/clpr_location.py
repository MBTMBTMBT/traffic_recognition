#
# Chinese License Plate Recognition - 2/4
# clpr_location.py  -- 基于边缘检测、数学形态学、彩色特征等定位方法
#

import cv2
import numpy as np
import os


def verifysizes(rotated_rect, err):
    #
    # 检查旋转矩形的面积和长宽比是否是合理的
    #
    aspect = 3.7  # 3.7,经验值=3.5；Chinese general car plate size: 440x140 aspect 3.142857; Spain: 4.7272
    min_aspect = aspect - aspect * err  # 0.4->2.22; 0.5->1.85; 0.6->1.48; 0.7->1.11; 0.8->0.74; 0.9->0.37
    max_aspect = aspect + aspect * err  # 0.4->5.18; 0.5->5.55; 0.6->5.92; 0.7->6.29; 0.8->6.66; 0.9->7.03

    # min_area = 15 * aspect * 15  # 15, 707.12999
    min_area = 10 * aspect * 10  # 10, 370 （4_0.png）
    max_area = 148 * aspect * 148  # 125, 49106.24999
    rect_width, rect_height = rotated_rect[1]
    if rect_width == 0 or rect_height == 0:
        return False
    area = rect_width * rect_height
    aspect = rect_width / rect_height

    if aspect < 1:
        aspect = 1 / aspect

    # print("VS-1:min_area <= area <= max_area = ", int(min_area), int(area), int(max_area))
    # print("VS-2:min_aspect <= aspect <= max_aspect = ", int(min_aspect), int(aspect), int(max_aspect))
    if min_area <= area <= max_area and min_aspect <= aspect <= max_aspect:
        return True
    else:
        return False


def color_judge(card_imgs):
    #
    # 车牌颜色识别
    #     目前只识别蓝、绿、黄三色的车牌。可以排除一些不是车牌的矩形
    #
    colors = []
    plates = []
    for ck, card_img in enumerate(card_imgs):
        green = yellow = blue = 0
        card_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # cv2.imshow("cj-1: card_hsv", card_hsv)
        # cv2.waitKey(0)

        h, w = card_hsv.shape[:2]
        card_img_count = h * w

        # 确定车牌颜色
        for i in range(h):
            for j in range(w):
                H = card_hsv.item(i, j, 0)
                S = card_hsv.item(i, j, 1)
                V = card_hsv.item(i, j, 2)
                # print("cj-2: H, S, V = ", H, S, V)
                if 11 < H <= 34 and S > 34:  # 26<=H<=34, 34<=S;      11 < H <= 34
                    yellow += 1
                elif 35 < H <= 99 and S > 34:  # 35<=H<=77, 43<=S
                    green += 1
                elif 99 < H <= 124 and S > 34:  # 100<=H<=124, 43<=S
                    blue += 1

        maxcolor = max(yellow, green, blue)
        # print("cj-3: yellow, green, blue, card_img_count = ", yellow, green, blue, card_img_count)
        color = "no"
        if yellow * 2.2 >= card_img_count and yellow == maxcolor:  # 经验值：2
            color = "yellow"
        elif green * 3.1 >= card_img_count and green == maxcolor:  # 经验值：2.4, 京AD77972(3.03)
            color = "green"
        elif blue * 3.1 >= card_img_count and blue == maxcolor:  # 经验值：2.3, 蒙B099X8(3.03)
            color = "blue"

        if color != "no":
            colors.append(color)
            plates.append(card_img)
            # print("cj-4: color = ", color)
            # cv2.imshow("cj-4:card_img"+str(ck), card_img)
            # cv2.waitKey(0)
    return colors, plates


def location(source_img):
    #
    # 车牌定位
    #
    # 车牌定位的一个重要特点就是：车牌上有许多垂直边，假定是从正面看、车牌没有转动、没有透视变形。
    # 在查找垂直边之前，像将彩色图转换成灰度图，去除可能的噪声（由摄像机、其他环境噪声产生的）。

    # -----------------------------------------
    # -1- 整理图像尺寸。如果过大，则按比例缩小；反之，则放大
    #
    # 注意：-2-中的参数是按 MAX_WIDTH=500 时调试的。
    # 如果调大此数，需要重新调整参数！！！
    # 适当调大此数，可有助提高识别率
    MAX_WIDTH = 700  # 500 or 1000 *******
    rows, cols = source_img.shape[:2]
    if cols > MAX_WIDTH:  # 图片过大，则缩小
        change_rate = MAX_WIDTH / cols
        source_img = cv2.resize(source_img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    else:
        if cols < MAX_WIDTH:  # 图片过小，则放大
            change_rate = MAX_WIDTH / cols
            source_img = cv2.resize(source_img, (MAX_WIDTH, int(rows * change_rate)),
                                    interpolation=cv2.INTER_CUBIC)  # _CUBIC, _LINEAR, _NEAREST, _AREA, _LANCZOS4
    # cv2.imshow("L-1:source_img", source_img)
    # cv2.waitKey(0)

    # ---------------------------------------------
    # -2- 通过数学形态学运算和边缘检测算子找出所有特征轮廓
    #

    # 先高斯模糊，再灰度转换。
    gauss_img = cv2.GaussianBlur(source_img, (5, 5), 0)
    gauss_img = cv2.cvtColor(gauss_img, cv2.COLOR_BGR2GRAY)

    # 为了找到垂直边，使用Sobel过滤器并发现第一个水平导数。
    sobelx_img = cv2.Sobel(gauss_img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow("L-2:sobelx_img", sobelx_img)
    sobelx_img = cv2.convertScaleAbs(sobelx_img)
    # cv2.imshow("L-2:sobelx_img", sobelx_img)

    # Sobel过滤器之后，使用threshold过滤器获得一个二值图像。这里选用Otsu算法（需要8位输入图像），将自动决定最优阈值。
    ret, threshhold_img = cv2.threshold(sobelx_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("L-2:threshold_img", threshhold_img)

    # 通过应用闭合生态学操作，可以将每个垂直边线之间的空白空间移除，并将有着多个边的区域连接起来。
    # 在这一步，能够得到可能包含车牌的若干区域。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))  # 经验值(20,3), >= 17
    # print(kernel)
    close_img = cv2.morphologyEx(threshhold_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow("L-2:close_img", close_img)

    # 增加了图像腐蚀和膨胀的次数（经验值3次）；对于小图片（200，300左右）经验值2次
    kernel = np.ones((3, 3), np.uint8)
    close_img = cv2.erode(close_img, kernel, iterations=2)
    # cv2.imshow("L-2:erosion", close_img)
    close_img = cv2.dilate(close_img, kernel, iterations=2)
    # cv2.imshow("L-2:dilate", close_img)
    # cv2.waitKey(0)

    # 新加！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-31_0:no!-
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))  # 经验值(22,3),苏BD00008<=20
    # close_img = cv2.morphologyEx(close_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow("L-2:2close_img", close_img)
    # cv2.waitKey(0)

    # 至此，我们已经得到了图片中包含车牌的区域，然而绝大多数区域都不包含车牌。这些区域可以通过连接组件分析
    # 或使用findContours函数进行拆分。findContours函数用不同的方法和结果检索二值图像的轮廓。我们只需要
    # 获取具有任何层次关系的外部轮廓，和多边形逼近结果。
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 注意下面的cv2.drawContours在source_img原图上画轮廓线
    # cv2.drawContours(source_img, contours, -1, (0, 255, 0), 1)  # 画出所有绿色轮廓，最后一参数是线宽
    # cv2.imshow("L-26:contours_img", source_img)

    # 使用OpenCV的minAreaRect函数，对于每个检测到的轮廓，提取最小面积的边界矩形。该函数返回一个旋转矩形类。
    # 然后，在对每个区域进行分类之前，在每个轮廓上使用向量迭代器，得到旋转的矩形，并做一些初步的验证。这些基本
    # 验证是基于区域面积和宽高比。车牌的宽高比大约是440/140=3.142857并有
    # 40%的误差率。面积是以车牌的高度最小15像素、最大125像素计算的。
    # 注意：这些值的计算取决于图像尺寸和摄像机位置。
    # 中国国内主要车牌的尺寸（本程序只考虑了第一种尺寸）：
    # 440×140: 蓝底白字白框线, 小型汽车; 黄底黑字黑框线, 大型汽车前、教练汽车  3.142857
    # 220mm×140: 蓝底白字白框线, 轻便摩托车后                             1.571428
    # 220mm×95mm: 黄底黑字黑框线, 两、三轮摩托车前                         2.315789
    # 300mm×165mm: 黄底黑字黑框线, 农用运输车                             1.818181

    # --------------------
    # -3- 初步筛选出候选轮廓
    #
    # print("L-3:len(contours) = ", len(contours))
    candidate_contours = []  # 存储合理的轮廓
    for contour in contours:
        rot_rect = cv2.minAreaRect(contour)
        if verifysizes(rot_rect, err=1.2):  # 1.1, 0.9, 1.3, 苏BD00008(1.1), 11_0(1.2)
            candidate_contours.append(contour)
            # box = cv2.boxPoints(rot_rect)  # 以下四行只做调试用！！！！！！
            # box = np.int0(box)
            # cv2.drawContours(source_img, [box], 0, (0, 255, 0), 1)  # 注意：画框会影响后面颜色识别！！！！！！！
            # cv2.imshow("L-31:boxing-copy_img", source_img)
    # print("L-3:len(candidate_contours) = ", len(candidate_contours))
    # cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # -4- 将候选轮廓旋转成水平方向（倾斜度超过30度的舍弃），并截取彩色图保存在card_imgs中
    #
    i = 0
    h, w = source_img.shape[:2]
    card_imgs = []
    for contour in candidate_contours:
        rot_rect = cv2.minAreaRect(contour)
        x, y = rot_rect[0]
        width, height = rot_rect[1]
        angle = abs(rot_rect[2])
        # print("L-4:x, y, width, height, angle = ", int(x), int(y), int(width), int(height), int(angle))
        if angle == 90:
            angle = 0
        if width >= height:  # 顺时针旋转（负值）
            if angle > 30:
                continue
            angle = angle * (-1)
        else:  # 逆时针旋转（正值）
            width, height = height, width
            if angle != 0:  # new added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                angle = 90 - angle
            if angle > 30:
                continue

        binary_img = np.zeros((h, w), dtype=np.uint8)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        cv2.drawContours(binary_img, [box], 0, (255, 255, 255), 1)
        # cv2.imshow("L-4:binary_img"+str(i), binary_img)
        # 对矩形进行抗扭斜矫正
        rotate = cv2.getRotationMatrix2D((x, y), angle, 1)  # 获得仿射变化矩阵。angel正值表示逆时针旋转
        card_img = cv2.warpAffine(source_img, rotate, (w, h))  # 进行仿射变化
        # cv2.imshow("L-4:warpAffine-card_img"+str(i), card_img)
        binary_img = cv2.warpAffine(binary_img, rotate, (w, h))  # 进行仿射变化
        # cv2.imshow("L-4:warpAffine-binary_img"+str(i), binary_img)
        # cv2.waitKey(0)

        _cont, _hier = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("L-4:len(_cont) = ", len(_cont))
        for cnt in _cont:
            new_rect = cv2.minAreaRect(cnt)
            if verifysizes(new_rect, err=1.2):  # 0.9, 1.3, 11_0(1.2)
                new_x, new_y = new_rect[0]
                new_width, new_height = new_rect[1]
                # print("L-4:new_x, new_y, new_width, new_height = ", new_x, new_y, new_width, new_height)
                if new_width < new_height:
                    new_width, new_height = new_height, new_width
                card_img = cv2.getRectSubPix(card_img, (int(new_width), int(new_height)), (new_x, new_y))
                # cv2.imshow("L-4:card_img"+str(i), card_img)
                '''
                # 统一车牌输出的尺寸大小(144,33)(105, 33)。最优尺寸???????????
                card_img = cv2.resize(card_img, (148, 40), interpolation=cv2.INTER_AREA)  # img,(列,行);(144,33)最优尺寸???
                # cv2.imshow("L-4:resize-card_img"+str(i), card_img)
                # cv2.waitKey(0)
                '''
                card_imgs.append(card_img)
        i += 1
    # cv2.waitKey(0)

    # ------------------------------------------------------------------
    # -5- 对card_imgs（plate）中候选图判断颜色，只有蓝色、黄色和绿色记入colors
    #
    colors, plates = color_judge(card_imgs)
    # print("L-5:len(plates), colors = ", len(plates), colors)

    k = 0
    while k < len(plates):
        # cv2.imshow("L-5:pic"+str(k), plates[k])
        k += 1
    # cv2.waitKey(0)

    # -------------------------
    # -6- 用颜色进一步定位候选车牌
    #
    i = 0
    plate_imgs = []
    plate_colors = []
    for plate in plates:
        plate_copy = plate.copy()
        # cv2.imshow("L-6:plate"+str(i), plate)
        # cv2.waitKey(0)

        # 高斯模糊，转HSV色彩空间
        img = cv2.GaussianBlur(plate, (5, 5), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow("L-6:hsv", hsv)

        # 设定蓝色、黄色、绿色的阈值
        lower = 100
        upper = 130
        if colors[i] == "blue":
            lower = np.array([100, 43, 46])    # 模糊范围:[100,  43,  46][100, 110, 110]
            upper = np.array([124, 255, 255])  # 模糊范围:[124, 255, 255][130, 255, 255] 不能小于112！！！
        elif colors[i] == "yellow":
            lower = np.array([18, 34, 46])    # 模糊范围:[26,  43,  46][15, 55, 55] 不能小于19!!!
            upper = np.array([34, 255, 255])  # 模糊范围:[34, 255, 255][50, 255, 255]
        elif colors[i] == "green":
            lower = np.array([35, 43, 46])    # 模糊范围:[35,  43,  46]
            upper = np.array([110, 255, 255])  # 模糊范围:[77, 255, 255]

        # 根据阈值构建掩模
        mask = cv2.inRange(hsv, lower, upper)
        # cv2.imshow('L-6:inRange'+str(i), mask)

        # 用cv2.countNonZero统计mask中白点的数量nz，计算白点在mask中的占比。mask尺寸：148x40=5920
        # 占比>=0.91：不处理，直接返回；占比<=0.91：继续图像处理，并做findContours
        nz = cv2.countNonZero(mask)
        # print("L-6:nz, nz/5920 = ", nz, nz/5920)

        # 本程序的两个正常出口之一。不需要用颜色再做定位，可以进行分割了。
        if nz/5920 > 0.91:  # 此图不需要处理，next one （如果此前截图不全，将影响后面分割和识别。是不是不转？？？？）
            # print("L-6:no color locating needed further. go to next one.")

            # 统一车牌输出的尺寸大小(144,33)
            plate = cv2.resize(plate, (148, 40), interpolation=cv2.INTER_AREA)
            # cv2.imshow("L-6:A-resize" + str(i), plate)
            # cv2.waitKey(0)

            plate_imgs.append(plate)
            plate_colors.append(colors[i])
            i += 1
            continue
        # 否则，继续继续下面处理

        # 对原图像和掩模进行位运算
        btwa = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('L-6:bitwise_and'+str(i), btwa)

        # 灰度
        gray = cv2.cvtColor(btwa, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('L-6:gray'+str(i), gray)

        # 二值化
        ret, thre = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('L-6:thresh'+str(i), thre)

        # 闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))  # 黑E99999<=14(.904),渝AN7968>=12,
        binary = cv2.morphologyEx(thre, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('L-6:closed'+str(i), binary)
        # cv2.waitKey(0)

        # 经过上述处理后，在查找所有轮廓，有应该不多了
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("L-6:len(contours) = ", len(contours))

        # 筛选候选轮廓，多数情况下应是一个轮廓
        candidate_contours = []
        for contour in contours:
            rot_rect = cv2.minAreaRect(contour)
            if verifysizes(rot_rect, err=1.2):  # 0.4,   img0==>err=0.9, 11_2(1.2)
                candidate_contours.append(contour)
                # box = cv2.boxPoints(rot_rect)
                # box = np.int0(box)
                # cv2.drawContours(plate, [box], 0, (0, 0, 255), 1)
                # cv2.imshow("L-6:plate_img", plate)
        # print("L-6:len(candidate_contours) = ", len(candidate_contours))
        # cv2.waitKey(0)

        # 如果没有轮廓len(candidate_contours) == 0，则舍弃；否则继续下面处理
        if len(candidate_contours) == 0:
            # print("L-6:len(candidate_contours)==0, not a true plate. go to next one.")
            i += 1
            continue
        # 否则，len(candidate_contours) == 1，继续下面处理

        rot_rect = cv2.minAreaRect(candidate_contours[0])

        # 对绿色车牌，上调上边线
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        if colors[i] == "green":  # 对绿色车牌，上调上边线
            box[1][1] = box[1][1] - 4
            if box[1][1] < 0:
                box[1][1] = 0
            box[2][1] = box[2][1] - 4
            if box[2][1] < 0:
                box[2][1] = 0
            # print("box = ", box, box[1][1], box[2][1])
        # cv2.drawContours(plate, [box], 0, (0, 0, 255), 1)
        # cv2.imshow("L-6:box-plate_img", plate)

        cv2.drawContours(binary, [box], 0, (255, 255, 255), 1)  # 在binary上画白框 *************************
        # cv2.imshow("L-6:box-plate_img"+str(i), binary)
        # cv2.waitKey(0)

        _cont, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("L-6:len(_cont) = ", len(_cont))
        cand_cont = []
        for _ct in _cont:
            rot_rect = cv2.minAreaRect(_ct)
            if verifysizes(rot_rect, err=0.9):  # err=0.9
                cand_cont.append(_ct)
        # print("L-6:len(cand_cont) = ", len(cand_cont))

        # 如果没有轮廓len(cand_cont) == 0，则舍弃；否则继续下面处理
        if len(cand_cont) == 0:
            # print("L-6:it's impossible for len(cand_cont)==0. please set 'err' bigger !")
            i += 1
            continue

        rot_rect = cv2.minAreaRect(cand_cont[0])  # 虽然len(cand_cont)==1，rot_rect还是要重新执行获得

        x, y = rot_rect[0]
        width, height = rot_rect[1]
        angle = abs(rot_rect[2])
        # print("L-6:-x, y, width, height, angle = ", int(x), int(y), int(width), int(height), rot_rect[2])
        if angle == 90:
            angle = 0
        if width >= height:  # 顺时针旋转（负值）
             if angle > 30:
                i += 1
                continue
             angle = angle * (-1)
        else:  # 逆时针旋转（正值）
            width, height = height, width
            if angle != 0:  # new added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                angle = 90 - angle
            if angle > 30:
                i += 1
                continue
        # print("L-6:=x, y, width, height, angle = ", int(x), int(y), int(width), int(height), int(angle))

        # 对矩形进行抗扭斜矫正
        rotate = cv2.getRotationMatrix2D((x, y), angle, 1)  # 获得仿射变化矩阵。angel正值表示逆时针旋转
        plate = cv2.warpAffine(plate, rotate, (w, h))  # 进行仿射变化
        # cv2.imshow("L-6:warpAffine-plate"+str(i), plate)
        binary = cv2.warpAffine(binary, rotate, (w, h))  # 进行仿射变化
        # cv2.imshow("L-6:warpAffine-binary"+str(i), binary)
        # cv2.waitKey(0)

        cont, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("L-6:len(cont) = ", len(cont))
        ca_ct = []
        for ct in cont:
            rot_rect = cv2.minAreaRect(ct)
            if verifysizes(rot_rect, err=0.9):  # err=0.9
                ca_ct.append(ct)
        # print("L-6:len(ca_ct) = ", len(ca_ct))
        # cv2.imshow("L-6:ct-binary", binary)
        # cv2.waitKey(0)

        # 如果没有轮廓len(ca_ct) == 0，则舍弃；否则继续下面处理
        if len(ca_ct) == 0:
            # print("L-6:it's impossible for len(ca_ct)==0. please set 'err' bigger !")
            i += 1
            continue

        new_rect = cv2.minAreaRect(ca_ct[0])
        new_x, new_y = new_rect[0]
        new_width, new_height = new_rect[1]
        # print("L-6:new_x, new_y, new_width, new_height = ", new_x, new_y, new_width, new_height)
        if new_width < new_height:
            new_width, new_height = new_height, new_width

        plate = cv2.getRectSubPix(plate, (int(new_width), int(new_height)), (new_x, new_y))
        # cv2.imshow("L-6.1:getRectSubPix", plate)
        # cv2.waitKey(0)

        # 本程序的两个正常出口之二。用颜色再做定位后，就可以进行分割了。
        # 统一车牌输出的尺寸大小(144,33)(105, 33)。最优尺寸???????????
        plate = cv2.resize(plate, (148, 40), interpolation=cv2.INTER_AREA)  # img,(列,行);(144,33)最优尺寸???
        # cv2.imshow("L-6:B-resize"+str(i), plate)
        # cv2.waitKey(0)

        plate_imgs.append(plate)
        plate_colors.append(colors[i])

        i += 1
    # print("L-6:len(plate_imgs), len(plate_colors) = ", len(plate_imgs), len(plate_colors))

    if len(plate_imgs) == 0:
        # print("No any license plates are found :( -- clpr_location")
        # exit(2)
        return None, None
    else:
        # print("Total " + str(len(plate_imgs)) + " plates or plates-liked are found -- clpr_location")
        pass
    return plate_imgs, plate_colors


if __name__ == '__main__':
    # 读取源图像文件
    source_img = cv2.imdecode(np.fromfile("C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\masks\\train\\1\\7_3.png", dtype=np.uint8),
                              cv2.IMREAD_COLOR)  # 读有中文名的方法。cv2.imread()读中文名，报错

    # 车牌定位，分割，颜色识别
    card_imgs, colors = location(source_img)
    print("len(colors), colors = ", len(colors), colors)

    i = 0
    path = ".\\plates"
    for card_img in card_imgs:
        # cv2.imshow("card" + str(i), card_img)
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            # cv2.imwrite(path + "\\license_" + str(i) + "_" + colors[i] + ".jpg", card_img)
            print("imwrite OK !")
        except:
            print("imwrite NOT successful !")
        # cv2.imencode(".jpg", card_img)[1].tofile("license"+str(i)+".jpg")
        i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
