import numpy as np
from PIL import Image, ImageFont, ImageDraw


def get_progress_rate(present_num: int, total_num: int):
    # bar_num = present_num // total_num * 20
    # print("\b" * len(" %d / %d" % (present_num - 1, total_num)), end="")
    # print("%d / %d" % (present_num, total_num), end="")
    # string = "\b" * len(" %d / %d" % (present_num - 1, total_num))
    string = "%d / %d" % (present_num, total_num)
    percentage = present_num / total_num * 100
    percentage = "%.2f" % percentage
    # print(string, percentage)
    return string, percentage


def format_time(ms):
    ss = 1000
    mi = ss * 60
    hh = mi * 60

    hours = ms // hh
    minutes = (ms - hours * hh) // mi
    seconds = (ms - hours * hh - minutes * mi) // ss
    milliseconds = ms - hours * hh - minutes * mi - seconds * ss

    return "%dh/%dm/%ds/%dms" % (hours, minutes, seconds, milliseconds)


def put_chinese_string(cv_img, text: str, location: (), color: ()):
    # 加字库，支持显示中文字体
    font_path = "tools\\simsun.ttc"  # 宋体字体文件
    font = ImageFont.truetype(font_path, 24)  # 加载字体, 字体大小
    img_pil = Image.fromarray(cv_img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(location, text, font=font, fill=color)  # xy坐标, 内容, 字体, 颜色
    cv_img = np.array(img_pil)
    return cv_img
