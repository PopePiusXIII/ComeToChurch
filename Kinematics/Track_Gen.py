import numpy as np
import matplotlib.pyplot as plt


def track_gen(vx, ay, time_step):
    """Creates a track from vx, ay using v^2/r
    :param vx = km/h
    :param ay = g
    :param time_step = 1/(sampling frequency)
    :returns x_coord(m), y_coord(m), theta(rad), radius(m)
    """
    theta_sum = 0
    point = [[0], [0]]
    theta = [0.01]
    radius = [0.01]
    for i in range(0, len(vx), 1):
        if ay[i] == 0:  # avoid divide by 0
            ay[i] = 0.0001
        radius.append(((.277*vx[i])**2) / (9.806*ay[i]))     # v^2 / a = radius and .2777 km/h to m/s
        theta.append(.277*vx[i] * time_step / radius[i])   # theta = 2 pi * vx * timestep / (2*pi*r)
        point[0].append(radius[i] * np.cos(theta[-1] + theta_sum) + point[0][i] - radius[i]*np.cos(theta_sum))
        point[1].append(radius[i] * np.sin(theta[-1] + theta_sum) + point[1][i] - radius[i]*np.sin(theta_sum))
        theta_sum += theta[-1]
    return point[0], point[1], theta, radius


def track_gen_gps(gps_latitude, gps_longitude, limit):
    """
    Creates a track using GPS lat and long
    :param gps_latitude:  (deg)
    :param gps_longitude: (deg)
    :param limit: largest acceptable radius (m)
    :return: x_coord (m), y_coord (m), radius (m)
    """
    earth_radius = 6378100  # meters
    radius = []
    x0 = earth_radius * np.sin(np.pi / 2 - (gps_latitude[1] / 57.3)) * np.cos(gps_longitude[1] / 57.3)
    y0 = earth_radius * np.sin(np.pi / 2 - (gps_latitude[1] / 57.3)) * np.sin(gps_longitude[1] / 57.3)
    x = earth_radius * np.sin(np.pi / 2 - (gps_latitude / 57.3)) * np.cos(gps_longitude / 57.3) - x0
    y = earth_radius * np.sin(np.pi / 2 - (gps_latitude / 57.3)) * np.sin(gps_longitude / 57.3) - y0

    for i in range(0, len(x) - 20, 1):
        r = (radius_of_circle([x[i], y[i]], [x[i + 10], y[i + 10]], [x[i + 20],
                              y[i + 20]], limit))
        radius.append(r)

    return x, y, radius


def radius_of_circle(point1, point2, point3, limit):
    """
    Find center of cirlce by creating 2 line segments L12 and L23
    Then create perpendicular lines that pass through the middle of each line segment
    The intersection of these 2 lines is the center of the circle
    :param point1: (x, Y) (m)
    :param point2: (x, Y) (m)
    :param point3: (x, Y) (m)
    :param limit: max radius acceptable
    :return: calculated radius (m)
    """
    bipoint1 = np.array([[(point2[0] - point1[0])/2 + point1[0]], [(point2[1] - point1[1])/2 + point1[1]]])
    bipoint2 = np.array([[(point3[0] - point2[0])/2 + point2[0]], [(point3[1] - point2[1])/2 + point2[1]]])
    m1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
    m2 = (point3[1] - point2[1]) / (point3[0] - point2[0])
    a = np.array([[1/m1, 1], [1/m2, 1]])
    b = np.array([[bipoint1[0][0]/m1 + bipoint1[1][0]], [bipoint2[0][0]/m2 + bipoint2[1][0]]])
    center = np.linalg.solve(a, b)
    radius = ((point1[0] - center[0])**2 + (point1[1] - center[1])**2)**.5
    if radius > limit:
        radius = limit
    return radius


