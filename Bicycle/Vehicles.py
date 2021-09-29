import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as inte
import matplotlib.patches as patches
from matplotlib import animation
from Frame import Frame
from Frame import rotate2d


class Car:
    def __init__(self, frame_g):
        self.mass = 1500     # mass of car

        self.frame_a = Frame()    # local frame
        self.frame_a.v[0] = 20
        self.frame_g = frame_g    # non inertial frame for whole sim
        self.frame_ag = self.frame_a.transform(self.frame_g)    # veh with respect to non inertial frmae

        self.izz = 3500.0

        self.ics = np.array([self.frame_a.pos[0], self.frame_a.v[0], self.frame_a.pos[1], self.frame_a.v[1], self.frame_a.theta, self.frame_a.omega])

        self.result = {'f_fy': [],
                       'r_fy': [],
                       'steer': [],
                       'theta': [],
                       'beta': [],
                       'omega': [],
                       'alpha': [],
                       'f_sa': [],
                       'r_sa': [],
                       'time': [],
                       'vx': [],
                       'vy': [],
                       'ax': [],
                       'ay': [],
                       'rax': [],
                       'ray': [],
                       'x': [],
                       'y': []}

    class RearAxle:
        ky = 60000.0   # N/rad
        d = 2       # feet from axle centerline to cg m
        tire_width = .33    # m
        tire_diameter = 1   # m

    class FrontAxle:
        ky = 60000.0  # N/rad
        d = 2  # feet from axle centerline to cg m
        tire_width = .33  # m
        tire_diameter = 1  # m
        delta = 0.0   # steering angle rad

    def update_state(self, t, s):
        """
        :param t: time float
        :param s: np.array([self.pos[0], self.v[0], self.pos[1], self.v[1], self.theta, self.omega])
        :return:
        """
        self.frame_a.v = rotate2d(-self.frame_ag.theta, self.frame_ag.v)
        # print(self.frame_a.v)
        a11 = -(self.RearAxle.ky + self.FrontAxle.ky) / (self.mass * self.frame_a.v[0])
        a13 = (self.RearAxle.ky * self.RearAxle.d - self.FrontAxle.ky * self.FrontAxle.d) / (self.mass * self.frame_a.v[0]) - self.frame_a.v[0]
        a31 = -(self.FrontAxle.ky * self.FrontAxle.d - self.RearAxle.ky * self.RearAxle.d) / (self.izz * self.frame_a.v[0])
        a33 = -(self.FrontAxle.ky * self.FrontAxle.d**2 + self.RearAxle.ky * self.RearAxle.d**2) / \
               (self.izz * self.frame_a.v[0])

        a_mat = np.array([[a11, 0, a13],
                          [0, 0, 1],
                          [a31, 0, a33]], dtype=object)

        b11 = self.FrontAxle.ky / self.mass
        b31 = self.FrontAxle.ky * self.FrontAxle.d / self.izz
        b_mat = np.array([[b11], [0], [b31]])

        # update local state
        s = np.array([[self.frame_a.v[1]], [self.frame_a.theta], [self.frame_a.omega]], dtype=object)

        s_dot = np.matmul(a_mat, s) + np.matmul(b_mat, np.array([[self.FrontAxle.delta]]))

        self.frame_a.a[1] = s_dot[0][0]
        self.frame_a.omega = s_dot[1][0]
        self.frame_a.alpha = s_dot[2][0]

        # car frame
        self.frame_ag = self.frame_a.transform(self.frame_g)
        s_dot = np.array([self.frame_ag.v[0], self.frame_ag.a[0], self.frame_ag.v[1], self.frame_ag.a[1],
                          self.frame_ag.omega, self.frame_ag.alpha])

        kar.write_result(t, s_dot)
        return s_dot

    def write_result(self, t, s_dot):
        """
        used to log internal states before changing to global for integration
        :param t: current time
        :param s_dot: current state
        :return:
        """
        self.result['theta'].append(self.frame_a.v[1]/self.frame_a.v[0])
        self.result['omega'].append(s_dot[4])
        self.result['alpha'].append(s_dot[5])
        self.result['steer'].append(self.FrontAxle.delta)
        self.result['time'].append(t)
        self.result['x'].append(self.frame_a.pos[0])
        self.result['y'].append(self.frame_a.pos[1])
        self.result['vx'].append(self.frame_a.v[0])
        self.result['vy'].append(self.frame_a.v[1])


class Sim:
    def __init__(self, car):
        self.t_step = .001
        self.t0 = 0
        self.t1 = 30
        self.solution = inte.RK45(car.update_state, t0=0, y0=car.ics, t_bound=self.t1,
                                  vectorized=True, max_step=self.t_step)

        self.result_g = {
                       'time': [],
                       'ax': [],
                       'ay': [],
                       'vx': [],
                       'vy': [],
                       'x': [],
                       'y': [],
                       'theta': []}

        while self.solution.status == 'running':
            # car.FrontAxle.delta = np.cos(self.solution.t*.7) * .1
            car.FrontAxle.delta = .1
            self.solution.step()
            car.frame_ag.pos[0], car.frame_ag.v[0], car.frame_ag.pos[1], car.frame_ag.v[1], car.frame_a.theta, car.frame_a.omega = self.solution.y
            # rotate and return global
            self.result_g ['time'].append(self.solution.t)
            self.result_g ['x'].append(car.frame_ag.pos[0])
            self.result_g ['y'].append(car.frame_ag.pos[1])
            self.result_g ['ax'].append(car.frame_ag.a[0])
            self.result_g ['ay'].append(car.frame_ag.a[1])
            self.result_g ['vx'].append(car.frame_ag.v[0])
            self.result_g ['vy'].append(car.frame_ag.v[1])
            self.result_g ['theta'].append(car.frame_ag.theta)


frame_g = Frame()
kar = Car(frame_g)
kar.RearAxle.ky = kar.FrontAxle.ky
print("kys", kar.FrontAxle.ky, kar.RearAxle.ky)
sim = Sim(kar)

plt.scatter(sim.result_g ['x'], sim.result_g ['y'])
plt.show()

plt.scatter(sim.result_g ['ax'], sim.result_g ['ay'], c='r', marker='o', facecolor='none')
plt.show()

plt.plot(sim.result_g['ax'], c='b', marker='o')
plt.plot(sim.result_g['ay'], c='r', marker='o')
plt.show()

plt.scatter(sim.result_g ['time'], sim.result_g ['vx'], c='r', marker='x')
plt.scatter(sim.result_g ['time'], sim.result_g ['vy'], c='b', marker='x')
plt.scatter(sim.result_g ['time'], (np.array(sim.result_g ['vx']) ** 2 + np.array(sim.result_g ['vy'])**2)**.5, marker='+', c='k')

plt.scatter(kar.result['time'], kar.result['vx'], c='r', marker='x')
plt.scatter(kar.result['time'], kar.result['vy'], c='b', marker='x')
plt.scatter(kar.result['time'], (np.array(kar.result['vx']) ** 2 + np.array(kar.result['vy'])**2)**.5, marker='o', c='k')
plt.show()

plt.scatter(kar.result['time'], kar.result['theta'], c='k', marker='x')
plt.scatter(kar.result['time'], kar.result['alpha'], c='b')
plt.scatter(kar.result['time'], kar.result['omega'], c='g')
plt.show()

# Create the figure and axis
fig = plt.figure()
ax = plt.axes()

patch = patches.Rectangle((0, 0), kar.FrontAxle.d + kar.RearAxle.d, 2, fc='r')
patch1 = patches.Rectangle((kar.FrontAxle.d + kar.RearAxle.d, 2), kar.FrontAxle.tire_diameter, kar.FrontAxle.tire_width,
                           fc='k')
patches = [patch, patch1]
ax.set_xlim(-50, 50)
ax.set_ylim(0, 1000)
ax.scatter(sim.result['x'], sim.result['y'])
ax.axis('equal')


def init():
    ax.add_patch(patches[0])
    ax.add_patch(patches[1])
    return patches


def animate(i):
    ax.set_xlim(sim.result['x'][i] - 10, sim.result['x'][i] + 10)
    ax.set_ylim(sim.result['y'][i] - 10, sim.result['y'][i] + 10)

    patches[0].set_xy((sim.result['x'][i], sim.result['y'][i]))
    patches[0].angle = np.rad2deg(sim.result['theta'][i])

    patches[1].set_xy((sim.result['x'][i] + np.cos(sim.result['theta'][i]) * (kar.FrontAxle.d + kar.RearAxle.d),
                       sim.result['y'][i] + np.sin(kar.result['theta'][i]) * (kar.FrontAxle.d + kar.RearAxle.d)))
    patches[1].angle = np.rad2deg(kar.result['theta'][i] + kar.result['steer'][i])

    return patches


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(kar.result['x']), interval=10,blit=False)
plt.show()
