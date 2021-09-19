# -*- coding : utf-8 -*-
# coding: utf-8
import cv2
from tools import Display


def draw_line(background_img, name):
    global ix, iy, jx, jy, drawing, mode, img, copy, w_name
    drawing = False
    ix, iy = -1, -1
    jx, jy = -1, -1
    # img = np.zeros((512, 512, 3), np.uint8)
    img = background_img
    w_name = name
    # global ix, iy, jx, jy, drawing, mode, img
    # global img
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, draw_circle)
    copy = img.copy()
    while True:
        copy = Display.put_chinese_string(copy, "请依次点击两点画出一条线作为道路的中线", (5, 5), (0, 0, 255))
        if drawing:
            cv2.line(copy, (ix, iy), (x_glb, y_glb), (0, 0, 255))
            cv2.imshow(w_name, copy)
        cv2.imshow(name, copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            # break
            pass
        if ix != -1 and iy != -1 and jx != -1 and jy != -1:
            if ix != jx or iy != jy:
                # print(ix, iy, jx, jy)
                # cv2.line(img, (ix, iy), (jx, jy), (0, 0, 255))
                # cv2.waitKey(500)
                cv2.destroyWindow(name)
                return ix, iy, jx, jy


def draw_circle(event, x, y, flags, param):
    global ix, iy, jx, jy, drawing, mode, x_glb, y_glb
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        x_glb, y_glb = x, y
        cv2.line(copy, (ix, iy), (x, y), (0, 0, 255))
        cv2.imshow(w_name, copy)
        # cv2.line(copy, (ix, iy), (x, y), (0, 0, 255))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        jx, jy = x, y
        x_glb, y_glb = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # copy = img.copy()
            cv2.line(copy, (ix, iy), (x, y), (0, 0, 255))
            cv2.imshow(w_name, copy)
    # cv2.line(copy, (ix, iy), (x, y), (0, 0, 255))
    # cv2.imshow(w_name, copy)
    # print(ix, iy, jx, jy)
    if jx != -1:
        red = (0, 0, 255)
        cv2.line(copy, (ix, iy), (jx, jy), red)

# print(draw_line(None))
