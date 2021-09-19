#
# Chinese License Plate Recognition - 3/4
# clpr_segmentation.py  -- 基于轮廓查找方法，分割汉字、字母和数字
#

import cv2
import numpy as np


def verifychars(w, h, minwidth, maxwidth, maxheight):
    if w == 0 or h == 0:
        return False

    minheight = maxheight*0.4  # 20
    # maxwidth = 22  # 148/7 = 21.14
    # print("vc-1:minheight, h, maxheight = ", minheight, h, maxheight)
    # print("vc-2:minwidth, w, maxwidth = ", minwidth, w, maxwidth)
    if minheight <= h <= maxheight and minwidth < w <= maxwidth+3:
        return True
    else:
        return False


def contour_cutting(plate_img, thre_img, color):
    #
    # 分割字符
    #
    # 对定位后的候选车牌进行处理，将其中的字符（包括：汉字、字母和数字）分割提取出来。

    # -----------
    # -1- 预处理
    #
    # cv2.imshow("cc-1:0-plate_img", plate_img)
    # cv2.imshow("cc-1:0-thre_img", thre_img)
    orig_height, orig_width = thre_img.shape[:2]
    height = orig_height
    width = orig_width

    # 为了隔断螺钉与字符的连接，画上边界黑线
    cv2.line(thre_img, (0, 0), (width, 0), (0, 0, 0), 1)  # 2
    # cv2.imshow("cc-1:---thre_img", thre_img)
    # 为了隔断螺钉与字符的连接，画下边界黑线
    cv2.line(thre_img, (0, height-1), (width, height-1), (0, 0, 0), 1)  # 2
    # cv2.imshow("cc-1:___thre_img", thre_img)

    close_img = thre_img.copy()

    # 避免有些字符上下分开
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # 避免有些字符上下分开,(2, 4)
    close_img = cv2.morphologyEx(close_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow("cc-1:MORPH_CLOSE", close_img)
    # cv2.waitKey(0)

    # 新增。为防止接收的小图片中字符存在竖裂纹，做一点预处理。  图片分辨率较好时不用，因为有可能会引起字符粘连！
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # 经验值(2,1)  12_0,25_01(<=2,1)
    # close_img = cv2.morphologyEx(thre_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow("cc-1:___close_img", close_img)
    # cv2.waitKey(0)

    # 注意：下面用close_img去计算，用thre_img去截取！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    # print("cc-1:height, width = ", height, width)
    if color == "green":  # 绿色车牌是8位字符长
        plate_charnum = 8
    else:  # 蓝色和黄色车牌是7位字符长
        plate_charnum = 7
    # print("cc-1:number of charactors of this plate should be ", plate_charnum)

    # -----------------------------------
    # -2- 处理字符上下粘连的情况（如果有的话）
    #

    # 第一次查找所有轮廓
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("cc-2:len(contours) = ", len(contours))
    if len(contours) == 0:
        # print("Not a true plate. Go to next one.")
        chars = []
        return chars

    # 查找候选轮廓
    chravg_h = hh = c = 0  # chravg_h记录候选字符平均高度，为处理上下粘连情况备用
    chravg_y = yy = 0  # chravg_y记录候选字符平均y值，为处理上下粘连情况备用
    miny = height  # 记录候选字符最小y值，为处理上下粘连情况备用
    candidate_contours = []  # 存储合理的轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow("cc-2:0-plate_img", plate_img)

        # 允许两个字符粘连的情况发生（*2）: maxwidth=round(width/plate_charnum*2)
        if verifychars(w, h, minwidth=2, maxwidth=round(width/plate_charnum*2), maxheight=height):
            if y+h <= height:  # 把类字符块触底的刨掉
                c += 1  # 参与统计的类字符块个数
                hh += h
                yy += y
                if y < miny:
                    miny = y
            candidate_contours.append(contour)
            # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            # cv2.imshow("cc-2:0-plate_img", plate_img)
    # print("cc-2:len1-len(candidate_contours) = ", len(candidate_contours))
    if len(candidate_contours) == 0:
        # print("Not a true plate. Go to next one.")
        chars = []
        return chars
    if c != 0:
        chravg_h = hh / c
        chravg_y = yy / c
    else:
        # print("Not a true plate. Go to next one.")
        chars = []
        return chars
    # cv2.waitKey(0)

    # 如果len(candidate_contours) < plate_charnum-1-2，说明符合字符的字符块少于车牌字符数，应该不是车牌，返回空的chars
    if len(candidate_contours) < plate_charnum-1-2:  # 允许有个别字符粘连的可能性，太短就认为不是了，舍弃并返回
        # print("The number of charactors found is too little. May not be a true plate. Go to next one.")
        chars = []
        return chars
    else:  # 处理可能的字符粘连情况。  京NE1246，鲁LD9016，鲁Q521MZ，蒙A277EP
        c2 = int(height - chravg_y - chravg_h) - 1
        c1 = miny - 1
        # 防止c1和c2出界
        if c1 < 0:
            c1 = 0
        if c2 < 0:
            c2 = 0
        cv2.line(close_img, (0, c1), (width, c1), (0, 0, 0), 1)  # 上边线，经验值：2
        cv2.line(close_img, (0, height-c2), (width, height-c2), (0, 0, 0), 1)  # 下边线，经验值：2
        # cv2.imshow("cc-2:top&bottomlines-thre_img", close_img)
        # cv2.waitKey(0)

    # ----------------------------------------------------------------------------------------
    # -3- 处理两个字符粘连的情况（如果有的话）。当然，通常粘连只可能发生在“圆点”（汉字和第一个字母后）的右侧
    #

    # 第二次查找所有轮廓
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("cc-2:len(contours) = ", len(contours))
    if len(contours) == 0:
        # print("Not a true plate. Go to next one.")
        chars = []
        return chars

    # 查找候选轮廓
    candidate_contours = []  # 存储合理的轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow("cc-3:0-plate_img", plate_img)

        # 允许两个字符粘连的情况发生（*2）: maxwidth=round(width/plate_charnum*2)
        if verifychars(w, h, minwidth=2, maxwidth=round(width/plate_charnum*2), maxheight=height):
            if y+h < height:  # 把类字符块触底的刨掉
                c += 1  # 参与统计的类字符块个数
                hh += h
                yy += y
                if y < miny:
                    miny = y
            candidate_contours.append(contour)
            # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.imshow("cc-3:0-plate_img", plate_img)
    # print("cc-3:len1-len(candidate_contours) = ", len(candidate_contours))
    if len(candidate_contours) <= plate_charnum - 2:
        # print("Not a true plate. Go to next one.")
        chars = []
        return chars
    # cv2.waitKey(0)

    # 保存候选轮廓外包矩形尺寸数据
    charset = []
    for contour in candidate_contours:
        x, y, w, h = cv2.boundingRect(contour)
        charset.append([x, y, w, h])  # 保存候选轮廓外包矩形尺寸数据

    # 按x坐标排序（倒序）
    charset.sort(key=lambda e: e[0], reverse=True)
    # print("cc-3:charset = ", charset)

    # 估算字符的平均宽度，比实际计算（在粘连时）还准确一些
    avgw = width / (plate_charnum + 3)

    # 查找粘连的两个字符，如果找到，则用竖线隔开
    i = 0
    while i < plate_charnum - 2:
        x, y, w, h = charset[i]
        # print("cc-3:avgw, x, y, w, h = ", avgw, x, y, w, h)
        if avgw*2*0.7 < w < avgw*2*1.5:  # 找到了粘连的两个字符
            char_img = thre_img[y:y + h, x:x + w]
            nonzero = cv2.countNonZero(char_img)
            area = w * h
            # print("cc-3:nonzero, area, nonzero/area = ", nonzero, area, nonzero / area)
            if nonzero / area < 0.75:  # 经验值:0.65，防止是尾部的图形块
                # 在粘连的两个字符中间画一个黑色竖线隔开（在close_img上)
                cv2.line(close_img, (x+int(w/2), 0), (x+int(w/2), height - 1), (0, 0, 0), 1)
        i += 1
    # cv2.imshow("cc-3:adjoined", close_img)
    # cv2.waitKey(0)

    # ---------------
    # -4- 处理字母数字
    #

    # 第三次查找所有轮廓
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("cc-4:len(contours) = ", len(contours))

    # 再次查找候选轮廓
    candidate_contours = []  # 存储合理的轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 不允许两个字符粘连的情况发生（上面已经解决了）: maxwidth=round(width/plate_charnum)
        if verifychars(w, h, minwidth=2, maxwidth=round(width/plate_charnum), maxheight=height):  # 第一次仅筛选字母数字，不含汉字, =3
            candidate_contours.append(contour)
            # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # cv2.imshow("cc-4:1-plate_img", plate_img)
    # print("cc-4:2-len(candidate_contours) = ", len(candidate_contours))
    # cv2.waitKey(0)

    # 保存候选字符坐标尺寸
    chravg_h = hh = 0  # chravg_h记录候选字符平均高度，为下一段if语句中的else备用
    chravg_y = yy = 0  # chravg_y记录候选字符平均y值，为下一段if语句中的else备用
    miny = height  # 记录候选字符最小y值，为下一段if语句中的else备用
    char_rect = []
    for contour in candidate_contours:
        x, y, w, h = cv2.boundingRect(contour)
        hh += h
        yy += y
        if y < miny:
            miny = y
        char_rect.append([x, y, w, h])  # 保存候选轮廓外包矩形尺寸数据
        # cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.imshow("cc-2:plate_img", plate_img)
    chravg_h = hh / len(candidate_contours)
    chravg_y = yy / len(candidate_contours)
    # print("cc-4:plate_charnum, len(char_rect) = ", plate_charnum, len(char_rect))
    # cv2.waitKey(0)

    # 如果字符块还是少，则请人工处理吧
    if len(char_rect) < plate_charnum - 1 - 2:
        # print("cc-4:Segmentation problem. Please check")
        chars = []
        return chars

    # 按x坐标排序（倒序）
    char_rect.sort(key=lambda e: e[0], reverse=True)

    # 求字符之间的平均距离和候选字符的最大高度。从车牌后边往前查，如果后面矩形与前面的距离小于平均距离，
    # 就把后面的矩形移除。主要目的是把后面的非“1”竖线等图块去除。参与计算距离的矩形要少一些，把左边去掉
    # 一些（在汉字左边可能会有一些特殊图块存在），否则平均距离可能会被拉小，影响判断准确性。
    # 为了统一字符的高度，取最大高度，避免有些字符因种种原因（如车牌钉将字符隔断）轮廓取矮了。
    i = 0
    list_dist = []
    list_posi = []
    maxh = 0  # 候选字符的最大高度
    while i < min(plate_charnum-1, len(char_rect)):  # 计算短一些，否则平均距离可能会被拉小，影响判断。
        x, y, w, h = char_rect[i]
        list_posi.append(int(x + w/2))
        if i != 0:
            if h >= maxh:
                maxh = h
        i += 1
    i = 0
    sum_d = sum_w = 0  # 计算平均距离和平均宽度
    while i < min(plate_charnum-1, len(char_rect))-1:  # 计算短一些，否则平均距离可能会被拉小，影响判断。
        x, y, w, h = char_rect[i]
        k = list_posi[i] - list_posi[i + 1]
        if i != 0 and i != min(plate_charnum, len(char_rect))-1-1:  # 避免两端的极端值影响
            sum_d += k
            sum_w += w
        list_dist.append(k)
        i += 1
    avg_dist = int(sum_d / (min(plate_charnum, len(char_rect))-1-2))
    avg_w = int(sum_w / (min(plate_charnum, len(char_rect))-1-2))
    # print("cc-4:avg_dist, avg_w, list_dist, list_posi = ", avg_dist, avg_w, list_dist, list_posi)
    i = tag = 0
    while i < min(plate_charnum-1, len(char_rect)) - 1:  # i < len(char_rect) - 1
        if list_dist[i] <= int(avg_dist*0.8) or list_dist[i] > int(avg_dist*2.0):  # 经验值：*0.8, *2.0
            char_rect.pop(i)
            # print("cc-4:char_rect["+str(i)+"] was removed due to distance")
            tag = 1
            i += 1
            continue
        x, y, w, h = char_rect[i]
        # print("cc-4:x, y, w, h = ", x, y, w, h)
        # print("cc-4:avg_w, int(avg_w*0.6), w, int(avg_w*0.8)", avg_w, int(avg_w*0.5), w, int(avg_w*0.7))
        # 只检查后1位就可以了。再往前检查出错可能性较大，因为经验公式适应性不好:-(
        if i < 1 and avg_w*0.7 < w < int(avg_w*0.8):  # 排除比“1”宽且比平均宽度*0.8小的非字母数字 !!!!!!!!!!!!!!
            char_rect.pop(i)
            # print("cc-4:char_rect[" + str(i) + "] was removed due to charactor's width")
            tag = 1
        i += 1
    # print("cc-4:=len(char_rect) = ", len(char_rect))

    # 字母数字的个数超过车牌的字母数字的个数，检查车牌尾部是否有其他图案插入。如果上面已经截出尾部（tag==1），就不做下面这段了
    # 也有可能是汉字前面查存在的图块造成len(char_rect) > plate_charnum
    if tag == 0 and len(char_rect) > plate_charnum:  # 可能有非字母数字插入到char_rect列表中了，很可能是车牌尾部的图案
        x, y, w, h = char_rect[0]
        char_img = thre_img[y:y + h, x:x + w]  # 字符两边略微加宽，左0，右0
        # cv2.imshow("cc-4:-char"+str(i), char_img)
        # cv2.waitKey(0)
        nonzero = cv2.countNonZero(char_img)
        area = w * h  # w * h
        # print("cc-4:nonzero, area, nonzero/area = ", nonzero, area, nonzero/area)
        # 既不是数字“1”的情况。是“1”：nonzero/area > 0.9 and w < avg_w*0.4;
        # 也不是字母数字（除“1”以外的）的情况。是：nonzero/area < 0.75 and w > avg_w*0.8 and w < avg_w*1.3;
        if (nonzero/area <= 0.9 or w >= avg_w*0.4) \
                and (nonzero/area >= 0.75 or w <= avg_w*0.8 or w >= avg_w*1.3):
            char_rect.pop(0)

    # 至此，字母数字已经检查完毕，len(char_rect)应当至少不小于plate_charnum - 1，
    # 否则可能有字母数字漏掉了。此图片应是真车牌，但分割失败了，报错并处理下一个。
    if len(char_rect) < plate_charnum - 1:
        # print("It's a true plate, but doing segmentation failed. Please check later. Go to next one this time.")
        chars = []
        return chars

    char_imgs = []
    i = maxw = 0  # maxw为对字母数字统计的最大宽度
    while i < plate_charnum - 1:  # 这里处理字母数字，只有第一个汉字没有处理
        x, y, w, h = char_rect[i]
        # print("cc-4:-maxh, x, y, w, h = ", maxh, x, y, w, h)
        if w >= maxw:  # 求最大宽度
            maxw = w
        if y >= int(maxh - h):  # 调整y和h值
            y = y - int(maxh - h)
            h = maxh
        # print("cc-2:=maxh, x, y, w, h = ", maxh, x, y, w, h)
        char_img = thre_img[y:y + h, x-1:x + w + 1]  # 字符两边略微加宽，左1，右1；y+h+1
        # cv2.imshow("cc-4:=char"+str(i), char_img)
        # cv2.waitKey(0)

        char_imgs.append(char_img)
        i += 1

    # ------------
    # -5- 分割汉字
    #

    # 按前面计算的字母数字的平均上下限度c1和c2的高度范围来截取汉字这块；宽度按字母数字统计的最大宽度截取
    # print("c1, c2, height, height-c2, x, maxw, x-1-maxw-2, x-1 = ",
    # c1, c2, height, height-c2, x, maxw, x-1-maxw-2, x-1)
    char_img = thre_img[c1:height-c2, x-1-maxw-2:x-1]  # [0:height, 0:x-1]保留原始高度（上面汉字可能只截取了一小部分）

    # 消除汉字的上下空白（实际是黑色）---------------------------------------
    # cv2.imshow("cc-5:top-bottom-Hanzi", char_img)
    # cv2.waitKey(0)
    h, w = char_img.shape[:2]
    # print("cc-5:h, w = ", h, w)
    toplines = 0
    flag = False
    for i in range(h):
        blackcount = 0
        for j in range(w):
            if char_img[i, j] == 0:
                blackcount += 1
            else:
                flag = True
                break
        if flag:
            break
        # print("cc-5:-blackcount = ", blackcount)
        if blackcount == w:
            toplines += 1

    bottomlines = 0
    flag = False
    for i in range(h):
        blackcount = 0
        for j in range(w):
            if char_img[h - i - 1, j] == 0:
                blackcount += 1
            else:
                flag = True
                break
        if flag:
            break
        # print("cc-5:=blackcount = ", blackcount)
        if blackcount == w:
            bottomlines += 1
    # print("cc-5:toplines, bottomlines = ", toplines, bottomlines)

    char_img = char_img[toplines:h - bottomlines - 1, 0:w - 1]
    # cv2.imshow("cc-5:top-bottom-cut-Hanzi", char_img)
    # cv2.waitKey(0)
    # ---------------------------------------------------------------

    char_imgs.append(char_img)
    # cv2.imshow("cc-5:Hanzi-thre_img", char_img)
    # cv2.waitKey(0)

    # 把字符按正序放入chars中，并返回
    chars = []
    i = 0
    while i < plate_charnum:
        chars.append(char_imgs[plate_charnum - i - 1])
        i += 1

    return chars


if __name__ == '__main__':
    # 读取已分割好的彩色车牌图片
    plate_img = cv2.imdecode(np.fromfile(".\\plates\\license_64_green.jpg",
                                         dtype=np.uint8), cv2.IMREAD_COLOR)  # 读有中文名的方法。cv2.imread()读中文名会报错

    # 锐化处理; 转灰度图
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    card_img = cv2.filter2D(plate_img, -1, kernel=kernel)
    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    # 手动设置车牌颜色
    # color = "blue"
    color = "green"
    # color = "yellow"
    # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
    if color == "green" or color == "yellow":
        gray_img = cv2.bitwise_not(gray_img)

    # 新加高斯降噪，针对模糊不清的小车牌，但效果双重，不好的是字符更不清楚了，故不用
    # gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # 中值滤波器。主要用来处理图像中的椒盐现象
    # cv2.imshow("thre_img1", gray_img)

    # 二值化
    ret, thre_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("thre_img", thre_img)

    # 轮廓查找方法，另一种字符切割方法 - QMA
    char_imgs = contour_cutting(plate_img, thre_img, color)
    print("function contour_cutting() is finished")

    if len(char_imgs) == 0:
        print("No any charactors are found")
        exit(1)

    i = 0
    for char_img in char_imgs:
        cv2.imwrite(".\\characters\\" + str(i) + ".jpg", char_img)
        cv2.imshow("chars" + str(i), char_img)
        i += 1
    cv2.waitKey(0)
