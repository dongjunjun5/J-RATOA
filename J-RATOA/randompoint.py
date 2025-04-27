import random
import math


def random_points_in_circle(num_points, radius):
    random.seed(53)
    """Returns a list of num_points randomly distributed within a circle with radius."""
    points = []
    for i in range(num_points):
        r = radius * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2*math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))
    print(points)
    return points
