import numpy as np
import matplotlib.pyplot as plt


def track_plotter(vx, ay, time_step):
    theta_sum = 0
    point = [[0], [0]]
    theta = []
    radius = []
    for i in range(0, len(vx), 1):
        if ay[i] == 0:
            ay[i] = 0.01
        radius.append(((.277*vx[i])**2) / (9.806*ay[i]))     # v^2 / a = radius
        theta.append(.277*vx[i] * time_step / radius[i])   # theta = 2 pi * vx * timestep / (2*pi*r)
        print "theta",radius[i]
        point[0].append(radius[i] * np.cos(theta[-1] + theta_sum) + point[0][i] - radius[i]*np.cos(theta_sum))
        point[1].append(radius[i] * np.sin(theta[-1] + theta_sum) + point[1][i] - radius[i]*np.sin(theta_sum))
        theta_sum += theta[-1]
    plt.plot(point[0])
    plt.show()
    return point[0], point[1], theta, radius
