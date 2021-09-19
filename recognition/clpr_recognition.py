#
# Chinese License Plate Recognition - 4/4
# clpr_recognition.py
#

import cv2
import os
import numpy as np
import libsvm
from recognition import clpr_segmentation
from numpy.linalg import norm  # 线性代数的范数


provinces = ["zh_cuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", "zh_gui", "贵", "zh_gui1", "桂", "zh_hei", "黑",
             "zh_hu", "沪", "zh_ji", "冀", "zh_jin", "津", "zh_jing", "京", "zh_jl", "吉", "zh_liao", "辽", "zh_lu", "鲁",
             "zh_meng", "蒙", "zh_min", "闽", "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan", "陕", "zh_su",
             "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", "zh_xin", "新", "zh_yu", "豫", "zh_yu1", "渝", "zh_yue",
             "粤", "zh_yun", "云", "zh_zang", "藏", "zh_zhe", "浙"]


cardtype = {"blue": "蓝色牌照",
            "green": "绿色牌照",
            "yellow": "黄色牌照"}


def preprocess_hog(digits):
    bin_n = 16
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def ocr(card_imgs, colors):
    #
    # 识别车牌文字
    #
    SZ = 20  # 训练图片长宽
    PROVINCE_START = 1000

    # 装载数据 - 字母和数字
    if os.path.exists("mats\\svm.dat"):
        svm_a = cv2.ml.SVM_load("mats\\svm.dat")
    else:
        raise FileNotFoundError('mats\\svm.dat')

    # 装载数据 - 汉字
    if os.path.exists("mats\\svmchinese.dat"):
        svm_c = cv2.ml.SVM_load("mats\\svmchinese.dat")
    else:
        raise FileNotFoundError('mats\\svmchinese.dat')

    # libsvm 测试
    ch_lib_svm = libsvm

    predict_result = []
    part_cards_list = []
    roi = None
    card_color = None
    for index, color in enumerate(colors):
        if color in ("blue", "yellow", "green"):
            card_img = card_imgs[index]
            copy_card_img = card_img.copy()
            # cv2.imshow("o-1:original", card_img)

            # 锐化处理; 转灰度图
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
            card_img = cv2.filter2D(card_img, -1, kernel=kernel)
            # cv2.imshow("o-1:filter2d", card_img)
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("o-1:BGR2GRAY", gray_img)
            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yellow":
                gray_img = cv2.bitwise_not(gray_img)
                # cv2.imshow("o-1:bitwise_not", gray_img)

            # 二值化
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("o-1:threshold", gray_img)

            # 进行字符分割
            part_cards = clpr_segmentation.contour_cutting(copy_card_img, gray_img, color)  # copy_card_img原图
            # print("o-1:len(part_cards) = ", len(part_cards))

            # 如果len(part_cards) == 0，说明是假车牌 或 真车牌分割错误，不处理。跳转下一个
            if len(part_cards) == 0:
                continue

            # 识别汉字、字符和数字
            for num, part_card in enumerate(part_cards):
                part_card_old = part_card
                # 边缘填充。cv2.BORDER_CONSTANT，固定值添加边框，统一都填充0
                w = abs(part_card.shape[1] - part_card.shape[0]) // 2  # 左右填充的宽度
                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # 图片缩放（缩小）：20 x 20
                part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                # cv2.imshow("o-2:part_card", part_card)

                part_card = preprocess_hog([part_card])
                # print(o-3:part_card.shape[:2])

                if num == 0:  # 中国车牌第一个是汉字
                    (p1, p2) = svm_c.predict(part_card)  # 匹配样本
                    # print(p2)
                    character = provinces[int(p2[0]) - PROVINCE_START]
                    # print(o-4:character)
                else:  # 识别字母
                    (p1, p2) = svm_a.predict(part_card)  # 匹配样本
                    character = chr(int(p2[0]))
                    # print(p2)
                    # print(o-5:character)

                predict_result.append(character)

            part_cards_list = part_cards
            roi = card_img
            card_color = color
            break
    return predict_result, roi, card_color, part_cards_list  # 识别到的字符、定位的车牌图像、车牌颜色、以及初始分割出的车牌各个位
