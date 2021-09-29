import numpy as np
import math
from scipy import integrate as inte
import matplotlib.pyplot as plt


def rotate2d(theta, vect):
    """
    :param theta: angle theta to rotate in rad
    :param vect: vector to rotate
    :return: rotated vector
    """
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=object)
    return np.matmul(r, vect)


class Frame:
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.v = np.array([0.0, 0.0])
        self.a = np.array([0.0, 0.0])
        self.theta = 0.0
        self.omega = 0.0
        self.alpha = 0.0

    def transform(self, frameb):
        """
        :param frameb: frame to transform into
        :return: frame object with respect to frameb
        """
        frameab = Frame()

        frameab.theta = np.subtract(self.theta, frameb.theta)
        frameab.omega = np.subtract(self.omega, frameb.omega)
        frameab.alpha = np.subtract(self.alpha, frameb.alpha)

        frameab.pos = np.subtract(self.pos, frameb.pos)
        frameab.v = np.subtract(rotate2d(frameab.theta, self.v), frameb.v)

        # print(frameab.v)
        coriolis = np.cross(np.array([0, 0, self.omega]), frameab.v)
        frameab.a = np.add(rotate2d(frameab.theta, np.subtract(self.a, frameb.a)), coriolis[0:2])
        return frameab

#
# class simulation:
#     frame_a = Frame()
#     frame_g = Frame()
#     frame_ag = Frame()
#
#     def __init__(self):
#         print("instnat")
#
# sim = simulation()
#
#
# def funky(t, s):
#     mass = 100
#     force = 2
#     radius = 20
#     sim.frame_a.v[0] = 20
#     sim.frame_a.v[1] = 0
#     sim.frame_a.a[0] = 0
#     sim.frame_a.a[1] = mass/force
#     sim.frame_a.omega = sim.frame_a.v[0] / radius
#
#     sim.frame_ag = sim.frame_a.transform(sim.frame_g)
#     return np.array([sim.frame_ag.v[0], sim.frame_ag.a[0], sim.frame_ag.v[1], sim.frame_ag.a[1], sim.frame_ag.omega, sim.frame_ag.alpha])
#
#     # vx, ax, vy, ay, omega, alpha
#
# t_step = .01
# t0 = 0
# t1 = 15
# solution = inte.RK45(funky, t0=0, y0=np.array([20, 20, 20, 0, 0, 1]), t_bound=t1, vectorized=False, max_step=t_step)
# while solution.status == 'running':
#     solution.step()
#     sim.frame_ag.pos[0], sim.frame_ag.v[0], sim.frame_ag.pos[1], sim.frame_ag.v[1], sim.frame_a.theta, sim.frame_a.omega = solution.y
#     # plt.scatter(sim.frame_ag.pos[0], sim.frame_ag.pos[1])
#     plt.scatter(sim.frame_ag.v[0], sim.frame_ag.v[1], c='k')
#     plt.scatter(sim.frame_a.v[0], sim.frame_a.v[1], c='r')
# plt.axis('equal')
# # plt.xlim((-40, 40))
# # plt.ylim((-40, 40))
# plt.show()

