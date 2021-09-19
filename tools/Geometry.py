import math


# 这几个类大概就是结合了一些高中数学知识来处理一些几何问题，
# 写的都比较简单，但是调用量相当大
# 感觉还是很不错的，写完这些以后很多地方用起来都很方便


class Point(object):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_coord(self) -> ():
        return self.x, self.y

    def __str__(self):
        return "x: %d, y:%d" % (self.x, self.y)

    @staticmethod
    def distance(a, b) -> float:
        x1, y1 = a.get_coord()
        x2, y2 = b.get_coord()
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Line(object):

    def __init__(self, coord_a: (), coord_b: ()):
        self.ax = coord_a[0]
        self.ay = coord_a[1]
        self.bx = coord_b[0]
        self.by = coord_b[1]

    # 高中学的点斜式判断点是在上在下，哈哈
    def point_is_above(self, p_coord: ()):
        if self.ax == self.bx:
            return False
        elif p_coord[1] > p_coord[0] * (self.ay - self.by) / (self.ax - self.bx) \
                + (self.ax * self.by - self.bx * self.ay) / (self.ax - self.bx):
            # print("above")
            return True
        else:
            # print("under")
            return False

    def point_at_left(self, p_coord: ()):
        if self.ay == self.by:
            return True
        if self.ax == self.bx:
            return p_coord[0] <= self.ax
        elif self.ax < self.bx and self.ay < self.by:
            return self.point_is_above(p_coord)
        elif self.ax > self.bx and self.ay > self.by:
            return self.point_is_above(p_coord)
        else:
            return not self.point_is_above(p_coord)


class Rect(object):

    def __init__(self, coord_x: int, coord_y: int, width: int, height: int):
        self.location = Point(coord_x, coord_y)
        self.width = width
        self.height = height

    def size(self) -> int:
        return int(self.width * self.height)

    def get_coord(self) -> ():
        return int(self.location.x), int(self.location.y)

    def get_coord_opposite(self) -> ():
        return int(self.location.x + self.width), int(self.location.y + self.height)

    def __str__(self):
        return "coord: %s, w: %d, h: %d, size: %d" \
               % (str(self.location), self.width, self.height, self.size())

    def get_mid_point(self) -> Point:
        return Point(self.location.x + self.width // 2, self.location.y + self.height // 2)

    def has_rect_inside(self, rect):
        p1 = Point(rect.get_coord()[0], rect.get_coord()[1])
        p2 = Point(rect.get_coord_opposite()[0], rect.get_coord_opposite()[1])
        if Rect.has_inside(self, p1) and Rect.has_inside(self, p2):
            return True
        else:
            return False
        pass

    def get_area(self):
        return self.width * self.height

    @ staticmethod
    def has_inside(rect, point):
        ax, ay = rect.get_coord()
        dx, dy = rect.get_coord_opposite()
        x, y = point.get_coord()
        return ax <= x <= dx and ay <= y <= dy

    @ staticmethod
    def are_overlapping(rect1, rect2) -> bool:
        '''
            a <- width -> c
                |
                height
                    |
            b <- width -> d
        '''
        startX1 = rect1.get_coord()[0]
        startY1 = rect1.get_coord()[1]
        endX1 = startX1 + rect1.width
        endY1 = startY1 + rect1.height
        startX2 = rect2.get_coord()[0]
        startY2 = rect2.get_coord()[1]
        endX2 = startX2 + rect2.width
        endY2 = startY2 + rect2.height
        return not (endY2 < startY1 or endY1 < startY2 or startX1 > endX2 or startX2 > endX1)

    # 计算重叠度的，后来没有用上
    @ staticmethod
    def overlapping_ratio(rect1, rect2) -> ():
        a_x1, a_y1 = rect1.get_coord()
        a_x2 = a_x1 + rect1.width
        a_y2 = a_y1
        a_x3 = a_x1
        a_y3 = a_y1 + rect1.height
        b_x1, b_y1 = rect2.get_coord()
        b_x2 = b_x1 + rect2.width
        b_y2 = b_y1
        b_x3 = b_x1
        b_y3 = b_y1 + rect2.height
        width = min(a_x2, b_x2) - max(a_x1, b_x1)
        height = min(a_y3, b_y3) - max(a_y1, b_y1)
        area = width * height
        return area / rect1.area(), area / rect2.area()


if __name__ == '__main__':
    rect1 = Rect(160, 270, 600, 720)
    rect2 = Rect(200, 300, 104, 93)
    rect3 = Rect(100, 100, 30, 30)
    print(rect1)
    print(rect2)
    print(rect3)
    print("1 and 2 is overlapping:" + str(is_overlapping(rect1, rect2)))
    print("2 and 3 is overlapping:" + str(is_overlapping(rect2, rect3)))
    print("1 and 3 is overlapping:" + str(is_overlapping(rect1, rect3)))
