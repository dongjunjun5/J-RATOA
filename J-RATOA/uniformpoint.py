import math
import numpy as np
import matplotlib.pyplot as plt


def evenly_distributed_points_in_circle(num_points, radius):
    """Returns a list of num_points evenly distributed in a circle with radius."""
    points = []
    angle_increment = 2 * math.pi / num_points
    arc_length = radius * angle_increment
    for i in range(num_points):
        angle = math.pi/2 + i * angle_increment
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points


def generate_points(total_points, square_size, min_distance):
    np.random.seed(0)
    points = np.empty((0, 2))
    while len(points) < total_points/2:
        x = np.random.uniform(-square_size, square_size)
        y = np.random.uniform(-square_size, square_size)

        # 새로운 점이 기존의 모든 점과 일정 거리 이상 떨어져 있는지 확인
        if len(points) == 0 or (np.min(np.sqrt(np.sum((points - [x, y])**2, axis=1))) >= min_distance):
            points = np.vstack([points, [x, y]])
    return points


def evenly_distributed_points_16():
    x_p1, y_p1 = (-300, 150)
    x_p2, y_p2 = (-100, 150)
    x_p3, y_p3 = (-300, -150)
    x_p4, y_p4 = (-100, -150)
    x_p5, y_p5 = (300, 150)
    x_p6, y_p6 = (100, 150)
    x_p7, y_p7 = (300, -150)
    x_p8, y_p8 = (100, -150)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8)]
    points = [(x // 10, y // 10) for x, y in points]
    print(points)
    return points


def evenly_distributed_points_20():
    x_p1, y_p1 = (-300, 120)
    x_p2, y_p2 = (-300, -120)
    x_p3, y_p3 = (-100, 180)
    x_p4, y_p4 = (-100, 0)
    x_p5, y_p5 = (-100, -180)
    x_p6, y_p6 = (300, 120)
    x_p7, y_p7 = (300, -120)
    x_p8, y_p8 = (100, 180)
    x_p9, y_p9 = (100, 0)
    x_p10, y_p10 = (100, -180)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8),(x_p9, y_p9),(x_p10, y_p10)]
    points = [(x // 10, y // 10) for x, y in points]
    print(points)
    return points


def evenly_distributed_points_24():
    x_p1, y_p1 = (-300, 200)
    x_p2, y_p2 = (-100, 200)
    x_p3, y_p3 = (-300, 0)
    x_p4, y_p4 = (-100, 0)
    x_p5, y_p5 = (-300, -200)
    x_p6, y_p6 = (-100, -200)
    x_p7, y_p7 = (300, 200)
    x_p8, y_p8 = (100, 200)
    x_p9, y_p9 = (300, 0)
    x_p10, y_p10 = (100, 0)
    x_p11, y_p11 = (300, -200)
    x_p12, y_p12 = (100, -200)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8),(x_p9, y_p9),(x_p10, y_p10),(x_p11, y_p11),(x_p12, y_p12)]
    print(points)
    return points


def evenly_distributed_points_28():
    x_p1, y_p1 = (-300, 200)
    x_p2, y_p2 = (-300, 0)
    x_p3, y_p3 = (-300, -200)
    x_p4, y_p4 = (-100, 300)
    x_p5, y_p5 = (-100, 100)
    x_p6, y_p6 = (-100, -100)
    x_p7, y_p7 = (-100, -300)
    x_p8, y_p8 = (300, 200)
    x_p9, y_p9 = (300, 0)
    x_p10, y_p10 = (300, -200)
    x_p11, y_p11 = (100, 300)
    x_p12, y_p12 = (100, 100)
    x_p13, y_p13 = (100, -100)
    x_p14, y_p14 = (100, -300)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8),(x_p9, y_p9),(x_p10, y_p10),(x_p11, y_p11),(x_p12, y_p12),(x_p13, y_p13),(x_p14, y_p14)]
    print(points)
    return points


def evenly_distributed_points_32():
    x_p1, y_p1 = (-300, 300)
    x_p2, y_p2 = (-300, 100)
    x_p3, y_p3 = (-300, -100)
    x_p4, y_p4 = (-300, -300)
    x_p5, y_p5 = (-100, 300)
    x_p6, y_p6 = (-100, 100)
    x_p7, y_p7 = (-100, -100)
    x_p8, y_p8 = (-100, -300)
    x_p9, y_p9 = (300, 300)
    x_p10, y_p10 = (300, 100)
    x_p11, y_p11 = (300, -100)
    x_p12, y_p12 = (300, -300)
    x_p13, y_p13 = (100, 300)
    x_p14, y_p14 = (100, 100)
    x_p15, y_p15 = (100, -100)
    x_p16, y_p16 = (100, -300)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8),(x_p9, y_p9),(x_p10, y_p10),(x_p11, y_p11),(x_p12, y_p12),(x_p13, y_p13),(x_p14, y_p14),(x_p15, y_p15),(x_p16, y_p16)]
    print(points)
    return points


def evenly_distributed_points_4vs6():
    x_p1, y_p1 = (-300, 150)
    x_p2, y_p2 = (-100, 150)
    x_p3, y_p3 = (-300, -150)
    x_p4, y_p4 = (-100, -150)

    x_p5, y_p5 = (300, 200)
    x_p6, y_p6 = (100, 200)
    x_p7, y_p7 = (300, 0)
    x_p8, y_p8 = (100, 0)
    x_p9, y_p9 = (300, -200)
    x_p10, y_p10 = (100, -200)

    points = [(x_p1, y_p1),(x_p2, y_p2),(x_p3, y_p3),(x_p4, y_p4),(x_p5, y_p5),(x_p6, y_p6),(x_p7, y_p7),(x_p8, y_p8),(x_p9, y_p9),(x_p10, y_p10)]
    print(points)
    return points


def evenly_distributed_points_3vs7():
    x_p1, y_p1 = (-200, 200)
    x_p2, y_p2 = (-200, 0)
    x_p3, y_p3 = (-200, -200)

    x_p4, y_p4 = (300, 200)
    x_p5, y_p5 = (300, 0)
    x_p6, y_p6 = (300, -200)
    x_p7, y_p7 = (100, 300)
    x_p8, y_p8 = (100, 100)
    x_p9, y_p9 = (100, -100)
    x_p10, y_p10 = (100, -300)

    points = [(x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4), (x_p5, y_p5), (x_p6, y_p6), (x_p7, y_p7),
              (x_p8, y_p8), (x_p9, y_p9), (x_p10, y_p10)]
    print(points)
    return points


def evenly_distributed_points_2vs8():
    x_p1, y_p1 = (-200, 200)
    x_p2, y_p2 = (-200, -200)

    x_p3, y_p3 = (300, 300)
    x_p4, y_p4 = (300, 100)
    x_p5, y_p5 = (300, -100)
    x_p6, y_p6 = (300, -300)
    x_p7, y_p7 = (100, 300)
    x_p8, y_p8 = (100, 100)
    x_p9, y_p9 = (100, -100)
    x_p10, y_p10 = (100, -300)

    points = [(x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4), (x_p5, y_p5), (x_p6, y_p6), (x_p7, y_p7),
              (x_p8, y_p8), (x_p9, y_p9), (x_p10, y_p10)]
    print(points)
    return points


def evenly_distributed_points_1vs9():
    x_p1, y_p1 = (-200, 0)

    x_p2, y_p2 = (100, 350)
    x_p3, y_p3 = (100, 150)
    x_p4, y_p4 = (100, 0)
    x_p5, y_p5 = (100, -150)
    x_p6, y_p6 = (100, -350)
    x_p7, y_p7 = (300, 250)
    x_p8, y_p8 = (300, 80)
    x_p9, y_p9 = (300, -80)
    x_p10, y_p10 = (300, -250)

    points = [(x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4), (x_p5, y_p5), (x_p6, y_p6), (x_p7, y_p7),
              (x_p8, y_p8), (x_p9, y_p9), (x_p10, y_p10)]
    print(points)
    return points

# def plot_points_in_square(total_points, square_size, min_distance):
#     points = generate_points(total_points, square_size, min_distance)
#
#     # 그래프 그리기
#     plt.scatter(points[:, 0], points[:, 1])
#     plt.xlim(-square_size/2, square_size/2)
#     plt.ylim(-square_size/2, square_size/2)
#     plt.gca().set_aspect('equal', adjustable='box')  # 가로 세로 비율 동일하게 설정
#     plt.show()
#
#
# # 예시로 20개의 점을 100x100 정사각형 내에 최소 거리 10 유지하며 균등하게 분포시키기
# plot_points_in_square(24, 100, 16)