from __future__ import division
import numpy as np
import math
from collections import OrderedDict
from copy import deepcopy
# import matplotlib.animation
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from matplotlib import cm
from scipy.optimize import fsolve


def magnitude(point_a, point_b):
    """finds the distance between any two points"""
    vector = np.subtract(point_a, point_b)
    total = 0
    for i in vector:
        total += i**2
    mag = total ** .5
    return mag


def magnitude_vect(vect):
    """pass vect as list or array returns the magnitude"""
    mag = (vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2) ** .5
    return mag


def norm_vect(vect):
    """normalizes a given vector
    RETURNS A UNIT VECTOR"""
    mag = (vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2) ** .5
    return np.divide(vect, mag)


def vect_creator(point_a, point_b):
    """Returns Point a -  Point b"""
    vect = np.subtract(point_a, point_b)
    return vect


def two_d_vertical_angle(lower_point, upper_point):
    """finds the 2d vector angle from the vertical. arrays must be 1d len of 2 think xz, yz"""
    """finds angle from the vertical. It is good for scenarios such as camber"""
    vect = np.subtract(upper_point, lower_point)
    np.ndarray.tolist(vect)
    # project to front view by deleting x term
    # vertical vect
    vert_vect = [0, 1]

    # using this relation http://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    angle = np.arccos(np.divide(np.dot(vect, vert_vect), (magnitude(lower_point, upper_point)))) * 180 / math.pi
    return angle


def two_d_horizontal_angle(lower_point, upper_point):
    """finds the 2d vector angle from the horizontal"""

    """finds angle from the horizontal. It is good for scenarios such as jacking coeff and anti squat"""
    vect = np.subtract(upper_point, lower_point)
    np.ndarray.tolist(vect)
    # project to front view by deleting x term
    # vertical vect
    horiz_vect = [1, 0]

    # using this relation http://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    angle = np.arccos(np.divide(np.dot(vect, horiz_vect), (magnitude(lower_point, upper_point)))) * 180 / math.pi
    return angle


def vector_angle_finder(vect_1, vect_2):
    """returns the angle in degrees between any two 3d vectors.
    http://www.analyzemath.com/stepbystep_mathworksheets/vectors/vector3D_angle.html"""
    theta = np.arccos(np.dot(vect_1, vect_2) / (magnitude_vect(vect_1) * magnitude_vect(vect_2)))
    angle = theta * 180 / math.pi
    return angle


def three_point_method(d, point_a, point_b, point_c, point_a0, point_b0, point_c0, point_d0):
    """"finds the position of a 4th point(d) when given 3 points and distance from point d to all other points
    Be aware that final equation is nonlinear so 2 possible solutions
    page 44 "The Multibody Systems Approach by Mike Blundell and Damian Harty"""
    dx, dy, dz = d

    mag_rda = magnitude(point_a0, point_d0)     # length of vector RDA
    mag_rdb = magnitude(point_b0, point_d0)     # length of vector RDB
    mag_rdc = magnitude(point_c0, point_d0)     # length of vector RDC

    # set up equation for scipy.fsolve by adding all 3 together and moving all over to right side
    eq1 = (dx - point_a[0])**2 + (dy - point_a[1])**2 + (dz - point_a[2])**2 - mag_rda**2
    eq2 = (dx - point_b[0])**2 + (dy - point_b[1])**2 + (dz - point_b[2])**2 - mag_rdb**2
    eq3 = (dx - point_c[0])**2 + (dy - point_c[1])**2 + (dz - point_c[2])**2 - mag_rdc**2

    return eq1, eq2, eq3


def shortest_line_to_point(point_a, point_b, point_c):  # where a and b are on spin axis, c is the point spinning round
    """
    Where the math came from
    https://www.youtube.com/watch?v=9wznbg_aKOo method #2"""
    axis_vect = np.subtract(point_a, point_b)
    axis_mag = magnitude(point_a, point_b)
    unit_axis = np.divide(axis_vect, axis_mag)  # unit of pp
    #  pp' constants - p

    # pp dot u
    t = np.sum(np.dot(unit_axis, unit_axis))
    c = np.sum(np.dot(np.subtract(point_b, point_c), unit_axis))
    p = -c / t
    project_point_on_axis_add = (np.multiply(unit_axis, p))
    project_point_on_axis = project_point_on_axis_add + point_b
    distance = magnitude(point_c, project_point_on_axis)
    return distance, project_point_on_axis


def parametrized_circle(point_a, point_b, point_c, theta):
    """Rotates a point c around an axis AB
    POINT A AND B LIE ON SPIN AXIS POINT C IS POINT THAT ROTATES ABOUT AXIS AB"""
    radius, center = shortest_line_to_point(point_a, point_b, point_c)
    # print'center, radius \n', center, radius
    center_axis = np.subtract(point_a, point_b)
    # print 'center axis %s , radius %s, center %s' % (center_axis, radius, center)
    # center_axis dot <1,1,z> = 0 returns perp vector
    in_plane = norm_vect(np.subtract(point_c, center))
    perp_1 = np.cross(center_axis, in_plane)
    perp_2 = np.cross(center_axis, perp_1)
    # print 'perp dick', perp_1, perp_2
    # norm perpendicular vectors
    perp_1 = norm_vect(perp_1)
    perp_2 = norm_vect(perp_2)
    if -1e-6 > np.dot(perp_1, perp_2) > 1e-6 or -1e-6 > (np.dot(perp_1, center_axis)) > 1e-6 or \
       -1e-6 > np.dot(perp_2, center_axis) > 1e-6:
        print 'not perpendicular'
        # print np.dot(perp_1, perp_2), np.dot(perp_1, center_axis), np.dot(perp_2, center_axis)
    x = center[0] + (radius * math.cos(theta) * perp_2[0]) + (radius * math.sin(theta) * perp_1[0])
    y = center[1] + (radius * math.cos(theta) * perp_2[1]) + (radius * math.sin(theta) * perp_1[1])
    z = center[2] + (radius * math.cos(theta) * perp_2[2]) + (radius * math.sin(theta) * perp_1[2])
    return [x, y, z]


def double_bearing_link(point_a, point_b, point_c, orig_point_d, new_point_d, lower_bound, upper_bound):
    """Solves for upper point after step around ab.
    solve for new theta of an upper link from movement of lower link
    find theta of other link that preserves the links length (link=pushrod/upright/etc)

    point a and point_b lie on spin axis
    point c and d are on the link static pos (IF THERE IS A ROCKER THE LINK TO ROCKER POINT MUST BE C)

    THEORY: LINK DE MUST MAINTAIN SAME LENGTH AFTER IT MOVES.
    """
    length = magnitude(point_c, orig_point_d)
    theta = bisection(magnitude, parametrized_circle, point_a, point_b, point_c, new_point_d, lower_bound, upper_bound,
                      length)
    return theta


def theta_finder(theta, point_a, point_b, point_c, point_c_new):
    """Using parameterized circle find theta so parameterized circle - (the point from 3 point method) = 0"""
    x, y, z = parametrized_circle(point_a, point_b, point_c, theta)
    residual = (x - point_c_new[0])**2 + (y - point_c_new[1])**2 + (z - point_c_new[2])**2
    return residual


def bisection(f, fu, point_a, point_b, point_c, point_d, lower_bound, upper_bound, length):
    """
    Takes the magnitude of different points on the parameterized circle until the length of the pushrod is preserved
     , start values [a,b], tolerance value(optional) TOL and max number of iterations(optional) NMAX and returns the
     root of the equation using the bisection method.

     NOTE IF SEARCHING FOR ANSWER BETWEEN 3.14 - > 6.28 FAILS TRY MAKING 6.28 THE LOWER BOUND!!!
    """
    n = 1
    theta = 0
    a = lower_bound
    b = upper_bound
    while n <= 100:
        theta = (a + b) / 2.0
        if -1e-6 < f(fu(point_a, point_b, point_c, theta), point_d) - length < 1e-6:
            # print 'Residual', f(fu(point_a, point_b, point_c, theta), point_d) - length
            # print 'iteration', n
            return theta
        else:
            n = n + 1
            if f(fu(point_a, point_b, point_c, theta), point_d) - length > 0:
                b = theta
            else:
                a = theta

    print 'failedtheta', theta, 'Residual', f(fu(point_a, point_b, point_c, theta), point_d) - length
    print 'iteration', n
    return False


def plane_equation(point_a, point_b, point_c):
    """Finds the normal unit vector of a plane"""
    v1 = np.subtract(point_a, point_c)
    v2 = np.subtract(point_a, point_b)
    normal = np.cross(v1, v2)
    # print 'b4 norm', normal
    unit_normal = norm_vect(normal)
    # print 'unityyy', unit_normal
    return unit_normal


def plot_plane(unit_normal, x_array, y_array, fore):
    """plots a plane given a normal vector and a point on the plane
    https://www.youtube.com/watch?v=0qYJfKG-3l8"""
    # print'unit normal = ', unit_normal
    z = (((unit_normal[0] * (fore[0] - x_array)) + (unit_normal[1] * (fore[1] - y_array))) / unit_normal[2]) + fore[2]
    # print 'plane numbers\n', z
    return z


def plot_line(unit_vect, point, array):
    """returns a list of x, y, z points on a line when given direction and a point on the line"""
    x_vals = []
    y_vals = []
    z_vals = []
    for i in array:
        x_vals.append(unit_vect[0] * i + point[0])
        y_vals.append(unit_vect[1] * i + point[1])
        z_vals.append(unit_vect[2] * i + point[2])

    return [x_vals, y_vals, z_vals]


def suspension_plot(ax, full_car_dict, planes_choice, instant_center_choice, *keys):
    """Plots the static suspension front or rear
    Choices: default false to display
    KWARGS: 'Left': Corner, 'Right': Corner
    EX: 'Left': Left Front"""
    # -------------------------Stack 3d Points To Form Links-----------------------------------
    left = keys[0]
    right = keys[1]
    # Left
    left_lower_control_arm = np.stack((full_car_dict[left]['Lower Fore'], full_car_dict[left]['Lower Out'],
                                       full_car_dict[left]['Lower Aft']), axis=-1)
    left_upper_control_arm = np.stack((full_car_dict[left]['Upper Fore'], full_car_dict[left]['Upper Out'],
                                       full_car_dict[left]['Upper Aft']), axis=-1)
    left_pushrod = np.stack((full_car_dict[left]['Pushrod Control Arm'], full_car_dict[left]['Pushrod Rocker']), axis=-1
                            )

    left_rocker = np.stack((full_car_dict[left]['Damper Rocker'], full_car_dict[left]['Rocker Pivot'],
                            full_car_dict[left]['Pushrod Rocker']), axis=-1)

    # Right
    right_lower_control_arm = np.stack((full_car_dict[right]['Lower Fore'], full_car_dict[right]['Lower Out'],
                                        full_car_dict[right]['Lower Aft']), axis=-1)
    right_upper_control_arm = np.stack((full_car_dict[right]['Upper Fore'], full_car_dict[right]['Upper Out'],
                                        full_car_dict[right]['Upper Aft']), axis=-1)

    right_pushrod = np.stack((full_car_dict[right]['Pushrod Control Arm'], full_car_dict[right]['Pushrod Rocker']),
                             axis=-1)
    heave_damper = np.stack((full_car_dict[right]['Damper Rocker'], full_car_dict[left]['Damper Rocker']), axis=-1)

    roll_damper_a = np.stack((full_car_dict[right]['Roll Damper a'], full_car_dict[left]['Roll Damper a']),
                             axis=-1)

    right_rocker = np.stack((full_car_dict[right]['Damper Rocker'], full_car_dict[right]['Rocker Pivot'],
                             full_car_dict[right]['Pushrod Rocker']), axis=-1)

    # Steering
    steering = np.stack((full_car_dict[left]['Tie Rod Upright'], full_car_dict[left]['Tie Rod Chassis'],
                         full_car_dict[right]['Tie Rod Chassis'], full_car_dict[right]['Tie Rod Upright']), axis=-1)

    # ---------------------Plane Instant Centers------------------------
    # LEFT
    left_uc_x = np.linspace(full_car_dict[left]['Upper Aft'][0], full_car_dict[left]['Upper Fore'][0], 10)
    left_uc_y = np.linspace(full_car_dict[left]['Upper Aft'][1], full_car_dict[left]['Upper Out'][1], 10)
    left_ucxx, left_ucyy = np.meshgrid(left_uc_x, left_uc_y)
    left_uc_plane = plane_equation(full_car_dict[left]['Upper Fore'], full_car_dict[left]['Upper Aft'],
                                   full_car_dict[left]['Upper Out'])
    left_uczz = plot_plane(left_uc_plane, left_ucxx, left_ucyy, full_car_dict[left]['Upper Fore'])

    left_lc_x = np.linspace(full_car_dict[left]['Lower Fore'][0], full_car_dict[left]['Lower Aft'][0], 10)
    left_lc_y = np.linspace(0, full_car_dict[left]['Lower Out'][1], 10)
    left_lc_plane = plane_equation(full_car_dict[left]['Lower Fore'], full_car_dict[left]['Lower Aft'],
                                   full_car_dict[left]['Lower Out'])
    left_lczz = plot_plane(left_lc_plane, left_lc_x, left_lc_y, full_car_dict[left]['Lower Fore'])
    left_lcxx, left_lcyy = np.meshgrid(left_lc_y, left_lc_x)

    # intersection line of two planes (instant center axis)
    left_ic_unit, left_ic_point = plane_intersection_line(left_uc_plane, left_lc_plane,
                                                          full_car_dict[left]['Upper Fore'],
                                                          full_car_dict[left]['Lower Fore'])
    middle_t = (full_car_dict[right]['Upper Fore'][0] - left_ic_point[0]) / left_ic_unit[0]
    left_intersection_line = plot_line(left_ic_unit, left_ic_point, np.linspace(middle_t - 20, middle_t + 30, 20))

    # RIGHT (same as left but with right points)
    # control arm planes
    right_uc_x = np.linspace(full_car_dict[right]['Upper Aft'][0], full_car_dict[right]['Upper Fore'][0], 10)
    right_uc_y = np.linspace(full_car_dict[right]['Upper Aft'][1], full_car_dict[right]['Upper Out'][1], 10)
    right_ucxx, right_ucyy = np.meshgrid(right_uc_x, right_uc_y)
    right_uc_plane = plane_equation(full_car_dict[right]['Upper Fore'], full_car_dict[right]['Upper Aft'],
                                    full_car_dict[right]['Upper Out'])
    right_uczz = plot_plane(right_uc_plane, right_ucxx, right_ucyy, full_car_dict[right]['Upper Fore'])

    right_lc_x = np.linspace(full_car_dict[right]['Lower Fore'][0], full_car_dict[right]['Lower Aft'][0], 10)
    right_lc_y = np.linspace(0, full_car_dict[right]['Lower Out'][1], 10)
    right_lc_plane = plane_equation(full_car_dict[right]['Lower Fore'], full_car_dict[right]['Lower Aft'],
                                    full_car_dict[right]['Lower Out'])
    right_lczz = plot_plane(right_lc_plane, right_lc_x, right_lc_y, full_car_dict[right]['Lower Fore'])
    right_lcxx, right_lcyy = np.meshgrid(right_lc_y, right_lc_x)

    # intersection line of two planes (instant center axis)
    right_ic_unit, right_ic_point = plane_intersection_line(right_uc_plane, right_lc_plane,
                                                            full_car_dict[right]['Upper Fore'],
                                                            full_car_dict[right]['Lower Fore'])
    middle_t = (full_car_dict[right]['Upper Fore'][0] - right_ic_point[0]) / right_ic_unit[0]
    right_intersection_line = plot_line(right_ic_unit, right_ic_point, np.linspace(middle_t - 30, middle_t + 30, 20))

    # -------------------------------Plot on Given Axis-----------------------------------------
    # set up axis and clear previous if animated
    ax.clear()
    ax.set_xlim(full_car_dict[left]['Upper Fore'][0] + 10, full_car_dict[left]['Upper Aft'][0] - 10)
    ax.set_ylim(-25, 25)
    ax.set_zlim(0, 25)
    ax.view_init(0, 0)
    # Tie rods
    ax.plot(steering[0], steering[1], steering[2], c='r')
    # Rockers
    ax.plot(left_rocker[0], left_rocker[1], left_rocker[2], c='k')
    ax.plot(right_rocker[0], right_rocker[1], right_rocker[2], c='k')
    # Control Arms
    ax.plot(left_lower_control_arm[0], left_lower_control_arm[1], left_lower_control_arm[2], c='b')
    ax.plot(left_upper_control_arm[0], left_upper_control_arm[1], left_upper_control_arm[2], c='b')
    ax.plot(right_lower_control_arm[0], right_lower_control_arm[1], right_lower_control_arm[2], c='b')
    ax.plot(right_upper_control_arm[0], right_upper_control_arm[1], right_upper_control_arm[2], c='b')
    # Pushrods
    ax.plot(left_pushrod[0], left_pushrod[1], left_pushrod[2], c='g')
    ax.plot(right_pushrod[0], right_pushrod[1], right_pushrod[2], c='g')
    # Dampers
    ax.plot(heave_damper[0], heave_damper[1], heave_damper[2], c='r')
    ax.plot(roll_damper_a[0], roll_damper_a[1], roll_damper_a[2], c='c')

    if planes_choice:   # simply looking for whether plane choice is true or not
        ax.plot(left_lower_control_arm[0], left_lower_control_arm[1], left_lower_control_arm[2], c='b')
        ax.plot(left_upper_control_arm[0], left_upper_control_arm[1], left_upper_control_arm[2], c='b')
        ax.plot(right_lower_control_arm[0], right_lower_control_arm[1], right_lower_control_arm[2], c='b')
        ax.plot(right_upper_control_arm[0], right_upper_control_arm[1], right_upper_control_arm[2], c='b')
        # Control Arm Planes
        ax.plot_surface(left_ucxx, left_ucyy, left_uczz)
        #  ax.plot_surface(left_lcyy, left_lcxx, left_lczz)
        ax.plot_surface(right_ucxx, right_ucyy, right_uczz)
        #  ax.plot_surface(right_lcyy, right_lcxx, right_lczz)

    # Instant center lines
    if instant_center_choice:   # simply looking for whether Instant Center choice is true or not
        ax.plot(*left_intersection_line)
        ax.plot(*right_intersection_line)


def wheel_center_disp_damper_movement(rocker_theta, dictionary, results_dictionary, heave):
    """This function returns the difference between a given (desired) wheel center displacement and the solved for
    guess given a damper displacement"""
    results_dictionary['Pushrod Rocker'][-1] = (parametrized_circle(dictionary['Rocker Pivot'],
                                                                    dictionary['Rocker Pivot Axis'],
                                                                    dictionary['Pushrod Rocker'],
                                                                    rocker_theta))
    results_dictionary['Damper Rocker'][-1] = (parametrized_circle(dictionary['Rocker Pivot'],
                                                                   dictionary['Rocker Pivot Axis'],
                                                                   dictionary['Damper Rocker'],
                                                                   rocker_theta))
    results_dictionary['Roll Damper a'][-1] = (parametrized_circle(dictionary['Rocker Pivot'],
                                                                   dictionary['Rocker Pivot Axis'],
                                                                   dictionary['Roll Damper a'],
                                                                   rocker_theta))
    results_dictionary['Pushrod Control Arm'][-1] = fsolve(three_point_method,
                                                           dictionary['Pushrod Control Arm'],
                                                           args=(dictionary['Upper Fore'],
                                                                 dictionary['Upper Aft'],
                                                                 results_dictionary['Pushrod Rocker'][-1],
                                                                 dictionary['Upper Fore'],
                                                                 dictionary['Upper Aft'],
                                                                 dictionary['Pushrod Rocker'],
                                                                 dictionary['Pushrod Control Arm']),
                                                           )

    results_dictionary['Lower Out'][-1] = fsolve(three_point_method, dictionary['Lower Out'],
                                                 args=(dictionary['Lower Fore'],
                                                       dictionary['Lower Aft'],
                                                       results_dictionary['Pushrod Control Arm'][-1],
                                                       dictionary['Lower Fore'],
                                                       dictionary['Lower Aft'],
                                                       dictionary['Pushrod Control Arm'],
                                                       dictionary['Lower Out']))

    results_dictionary['Upper Out'][-1] = fsolve(three_point_method, dictionary['Upper Out'],
                                                 args=(dictionary['Upper Fore'],
                                                       dictionary['Upper Aft'],
                                                       results_dictionary['Lower Out'][-1],
                                                       dictionary['Upper Fore'],
                                                       dictionary['Upper Aft'],
                                                       dictionary['Lower Out'],
                                                       dictionary['Upper Out']))

    # Steering
    results_dictionary['Tie Rod Upright'][-1] = fsolve(three_point_method,
                                                       dictionary['Tie Rod Upright'],
                                                       args=(results_dictionary['Tie Rod Chassis'][-1],
                                                             results_dictionary['Lower Out'][-1],
                                                             results_dictionary['Upper Out'][-1],
                                                             dictionary['Tie Rod Chassis'],
                                                             dictionary['Lower Out'],
                                                             dictionary['Upper Out'],
                                                             dictionary['Tie Rod Upright']))

    results_dictionary['Wheel Center'][-1] = (fsolve(three_point_method, dictionary['Wheel Center'],
                                                     args=(results_dictionary['Tie Rod Upright'][-1],
                                                           results_dictionary['Upper Out'][-1],
                                                           results_dictionary['Lower Out'][-1],
                                                           dictionary['Tie Rod Upright'],
                                                           dictionary['Upper Out'],
                                                           dictionary['Lower Out'],
                                                           dictionary['Wheel Center'])))

    # solved for wheel displacement - actual wheel displacement
    return (dictionary['Wheel Center'][2] - results_dictionary['Wheel Center'][-1][2] + heave)**2


def bump_sim(heave_list, steering_rack_disp_list, dictionary, motion, *args):
    """Solves all suspension positions with a given heave (in), steering rack displacement(in), and starting dictionary
    The heave and steering_rack_disp_list must be same length 1 d arrays
    RETURNS: dictionary with list of moved points the same length as motion
    chassis is stationary and wheel is moved. At the end the displacement at the wheel is moved to the chassis
    ARGS: 'Left Front' , 'Right Front', 'Right Rear', 'Left Rear' must be list even if one value"""
    results_dictionary = OrderedDict([('Left Front', OrderedDict([])),
                                     ('Right Front', OrderedDict([])),
                                     ('Left Rear', OrderedDict([])),
                                     ('Right Rear', OrderedDict([]))])

    first_iteration = True
    for heave, steering_rack_disp in zip(heave_list, steering_rack_disp_list):
        # establishing an empty dict for each corner to be over written with the results
        # fill dict each iteration with original, new values will overwrite when determined
        for corner_key in args:
            for key in dictionary[corner_key].keys():
                if first_iteration:
                    results_dictionary[corner_key][key] = []
                results_dictionary[corner_key][key].append(dictionary[corner_key][key])

            # bump is a one wheel scenario so must stop from iterating over keys.
            if motion == 'Bump' and corner_key != args[0]:
                break

            # solve for values using the wheel center disp movement method and overwrite corresponding values in results
            # fslove to minimize the residual of the function wheel center placement
            fsolve(wheel_center_disp_damper_movement, np.array([3.14]), args=(dictionary[corner_key],
                                                                              results_dictionary[corner_key], heave))
        first_iteration = False
    return results_dictionary


def ground_plane_shift(orig_points, sim_points):

    # ------------TRANSLATE WHEEL BACK TO GROUND AND MOVE CHASSIS POINTS----------------------
    for result in range(len(sim_points['Lower Aft'])):
        heave = orig_points['Wheel Center'][2] - sim_points['Wheel Center'][result][2]
        for key in sim_points.keys():
            sim_points[key][result] = [sim_points[key][result][0],
                                       sim_points[key][result][1],
                                       sim_points[key][result][2]+heave]


def sim_evaluation(orig_points, sim_points, motion, *corners):
    """evaluates sim results for camber, caster, etc
    pass keys of sim points as args(*corners) to be evaluated: TO CALC MOTION DEPENDENT ON TWO WHEELS PASS BOTH WHEELS
    """
    # defining dictionaries to store all the data for sim evaluations
    evaluations = OrderedDict([(
        'Camber Angle', []),
        ('Caster Angle', []),
        ('Motion Tire Heave(in)', []),
        ('Motion Body Heave(in)', []),
        ('Heave Damper Length(in)', []),
        ('Roll Damper Length(in)', []),
        ('Heave Damper Displacement(in)', []),
        ('Roll Damper Displacement(in)', []),
        ('Heave Heave Damper MR', []),
        ('Heave Roll Damper MR', []),
        ('Bump Heave Damper MR', []),
        ('Bump Roll Damper MR', [])])

    # each corner has its own set of evaluations
    post_eval_dict = OrderedDict([('Left Front', deepcopy(evaluations)),
                                  ('Right Front', deepcopy(evaluations)),
                                  ('Left Rear', deepcopy(evaluations)),
                                  ('Right Rear', deepcopy(evaluations))])

    if motion == 'Heave':    # temporary patch to differentiate between one wheel bump and heave
        for key in corners:
            ground_plane_shift(orig_points[key], sim_points[key])

    for key in corners:
        for i in range(0, len(sim_points[key][sim_points[key].keys()[0]]), 1):
            # -------------------------CAMBER---------------------------
            orig_camber = two_d_vertical_angle(orig_points[key]["Lower Out"][1:3], orig_points[key]["Upper Out"][1:3])
            # im using list slicing to choose only the points in yz
            post_eval_dict[key]['Camber Angle'].append(two_d_vertical_angle(sim_points[key]["Lower Out"][i][1:3],
                                                                            sim_points[key]["Upper Out"][i][1:3]) -
                                                       orig_camber)
            # -----------------------MOTION TIRE HEAVE------------------------
            post_eval_dict[key]['Motion Tire Heave(in)'].append(sim_points[key]['Wheel Center'][i][2] -
                                                                orig_points[key]['Wheel Center'][2])
            # -----------------------MOTION BODY HEAVE------------------------------
            post_eval_dict[key]['Motion Body Heave(in)'].append(sim_points[key]['Lower Fore'][i][2] -
                                                                orig_points[key]['Lower Fore'][2])

            # im using list slicing to choose only the points in yz
            post_eval_dict[key]['Caster Angle'].append(two_d_vertical_angle(sim_points[key]["Lower Out"][i][0:3:2],
                                                                            sim_points[key]["Upper Out"][i][0:3:2]))

            # -------------------DAMPER LENGTH-------------------------
            orig_heave_damper_length = magnitude(orig_points[corners[0]]['Damper Rocker'], orig_points[corners[1]]
                                                 ['Damper Rocker'])
            orig_roll_damper_length = magnitude(orig_points[corners[0]]['Roll Damper a'], orig_points[corners[1]]
                                                ['Roll Damper a'])

            post_eval_dict[key]['Heave Damper Length(in)'].append(magnitude(sim_points[corners[0]]['Damper Rocker'][i],
                                                                  sim_points[corners[1]]['Damper Rocker'][i]))
            post_eval_dict[key]['Roll Damper Length(in)'].append(magnitude(sim_points[corners[0]]['Roll Damper a'][i],
                                                                           sim_points[corners[1]]['Roll Damper a'][i]))
            # -----------------------DAMPER DISPLACEMENT------------------
            post_eval_dict[key]['Heave Damper Displacement(in)'].append(orig_heave_damper_length -
                                                                        post_eval_dict[key]['Heave Damper Length(in)']
                                                                        [i])
            post_eval_dict[key]['Roll Damper Displacement(in)'].append(orig_roll_damper_length -
                                                                       post_eval_dict[key]['Roll Damper Length(in)'][i])
            # ------------------------MOTION RATIO------------------------
            if motion == 'Heave':
                post_eval_dict[key]['Heave Heave Damper MR'].append(post_eval_dict[key]
                                                                    ['Heave Damper Displacement(in)'][-1] /
                                                                    post_eval_dict[key]['Motion Body Heave(in)'][-1])
                post_eval_dict[key]['Heave Roll Damper MR'].append(post_eval_dict[key]
                                                                   ['Roll Damper Displacement(in)'][-1] /
                                                                   post_eval_dict[key]['Motion Body Heave(in)'][-1])
            if motion == 'Bump':
                post_eval_dict[key]['Bump Heave Damper MR'].append(post_eval_dict[key]
                                                                   ['Heave Damper Displacement(in)'][-1] /
                                                                   post_eval_dict[key]['Motion Tire Heave(in)'][-1])
                post_eval_dict[key]['Bump Roll Damper MR'].append(post_eval_dict[key]
                                                                  ['Roll Damper Displacement(in)'][-1] /
                                                                  post_eval_dict[key]['Motion Tire Heave(in)'][-1])
        print post_eval_dict
    return post_eval_dict


def scatter_plot(x_axis, y_axis, fit=False, power=3, *args):
    print x_axis
    print y_axis
    if fit:
        poly = np.polyfit(x_axis, y_axis, power)
        plt.plot(x_axis, np.polyval(poly, x_axis),
                 label="{three:5.2f}x^3+{two:5.2f}x^2+{one:5.2f}x+{c:5.2f}".format(three=poly[0],
                                                                                   two=poly[1], one=poly[2], c=poly[3]))
    plt.scatter(x_axis, y_axis)
    plt.xlabel(args[0])
    plt.ylabel(args[1])
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()
    return


def three_d_vector_plane_intersection(point_a, point_b, point_c, point_d, point_e):
    """Finds the intersection point between vector and a plane noted by three points on plane CDE and vector AB
        video for math https://www.youtube.com/watch?v=qVvvy5hsQwk and
        https://www.youtube.com/watch?v=LSceoFSJ-f0
        INSTRUCTIONS:
        POINT_A - point on axis
        POINT_B - point on axis
        POINT_C - point on plane to be pierced
        POINT_D - point on plane to be pierced
        POINT_E - point on plane to be pierced
        """
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    nv = plane_equation(point_c, point_d, point_e)
    t = (nv[0] * c[0] + nv[1] * c[1] + nv[2] * c[2] - nv[0] * a[0] - nv[1] * a[1] - nv[2] * a[2]) / \
        (nv[0] * (b[0] - a[0]) + nv[1] * (b[1] - a[1]) + nv[2] * (b[2]-a[2]))
    x = a[0] + t * (b[0] - a[0])
    y = a[1] + t * (b[1] - a[1])
    z = a[2] + t * (b[2] - a[2])
    intersection = np.array([x, y, z])
    return intersection


def plane_intersection_line(plane_a, plane_b, point_a, point_b):
    # plane equation a(x-x0) + b(y-y0) + c(z-z0) = 0 or ax + by + cz = d

    try:
        da = np.matrix.dot(plane_a, point_a)
        db = np.matrix.dot(plane_b, point_b)

        # assumption logic to defer from singular matrix by adjusting guess from x=0  to y=0 to z=0. 2 of 3 have to = 0
        # x=0
        vector_direction = np.cross(plane_a, plane_b)
        b = np.array([[da], [db]])

        a = [[plane_a[1], plane_a[2]], [plane_b[1], plane_b[2]]]

        solution = np.linalg.solve(a, b)
        solution = [0, solution[0, 0].tolist(), solution[1, 0].tolist()]
        return vector_direction, solution

    except np.linalg.linalg.LinAlgError:
        # assumption logic to defer from singular matrix by adjusting guess from x=0  to y=0 to z=0. 2 of 3 have to = 0
        # z = 0 guess
        print 'some kind of bullshit matrix error'
        da = np.matrix.dot(plane_a, point_a)
        db = np.matrix.dot(plane_b, point_b)

        vector_direction = np.cross(plane_a, plane_b)
        b = np.array([[da], [db]])

        a = [[plane_a[0], plane_a[1]], [plane_b[0], plane_b[1]]]

        solution = np.linalg.solve(a, b)

        solution = [solution[0].tolist(), solution[1].tolist(), 0]
        return vector_direction, solution


def jacking_calculations(dictionary, view):
    """This function calculates instant centers and jacking coefficients for all four corners of the car.
    view is either "Front" or "Side"
    Returns values in the order, lf, rf, lr, RR"""
    f_list = []
    s_list = []
    corners = ['Left Front', 'Right Front', 'Left Rear', 'Right Rear']
    for corner in corners:
        # Establishing Instant Center Left Front
        ic_direction, ic_point = plane_intersection_line(
            plane_equation(dictionary[corner]['Upper Fore'],
                           dictionary[corner]['Upper Aft'],
                           dictionary[corner]['Upper Out']),
            plane_equation(dictionary[corner]['Lower Fore'],
                           dictionary[corner]['Lower Aft'],
                           dictionary[corner]['Lower Out']),
            dictionary[corner]['Upper Fore'],
            dictionary[corner]['Lower Fore'])
        axis = plot_line(ic_direction, ic_point, np.linspace(0, 2, 2))
        # Establishing Side View Instant Center
        ic_xz = three_d_vector_plane_intersection((axis[0][0], axis[1][0], axis[2][0]),
                                                  (axis[0][1], axis[1][1], axis[2][1]),
                                                  dictionary[corner]['Wheel Center'],
                                                  np.add(np.array(dictionary[corner]
                                                                  ['Wheel Center']), np.array([1, 0, 0])),
                                                  np.add(np.array(dictionary[corner]
                                                                  ['Wheel Center']), np.array([0, 0, 1])))
        # Establishing Front View Instant Center
        ic_yz = three_d_vector_plane_intersection((axis[0][0], axis[1][0], axis[2][0]),
                                                  (axis[0][1], axis[1][1], axis[2][1]),
                                                  dictionary[corner]['Wheel Center'],
                                                  np.add(np.array(dictionary[corner]
                                                                  ['Wheel Center']), np.array([0, 1, 0])),
                                                  np.add(np.array(dictionary[corner]
                                                                  ['Wheel Center']), np.array([0, 0, 1])))
        # Establishing Jacking Height
        y_val = dictionary['Performance Figures']['Center of Gravity'][1]
        cg_plane_points = [[1, y_val, 1], [-1, y_val, 4], [-3, y_val, 6]]
        wheel_center_ground = [(dictionary[corner]['Wheel Center'][0]),
                               (dictionary[corner]['Wheel Center'][1]), 0]
        np.array(wheel_center_ground)
        jacking_height = three_d_vector_plane_intersection(wheel_center_ground,
                                                           ic_yz, cg_plane_points[0], cg_plane_points[1],
                                                           cg_plane_points[2])
        # Establishing Jacking Coefficient
        wc_jh = np.subtract(jacking_height, wheel_center_ground)
        jacking_coeff = -abs(wc_jh[2] / wc_jh[1])
        # Establishing Pitch Coefficient
        wc_icxz = np.subtract(ic_xz, dictionary[corner]['Wheel Center'])
        wc_cg = np.subtract(dictionary['Performance Figures']['Center of Gravity'],
                            dictionary[corner]['Wheel Center'])
        pitch_coeff = (wc_icxz[2] / wc_icxz[0]) / (wc_cg[2] / wc_cg[0])
        if view == 'Front':
            f_list.append(jacking_coeff)
        elif view == 'Side':
            s_list.append(pitch_coeff)
        else:
            print 'Wtf, you want an isometric or something?'
            return
    if view == 'Front':
        return f_list
    elif view == 'Side':
        return s_list
    else:
        print 'view does not equal Front or Side'
        return


def four_corner_wheel_displacement(dictionary, lat_force, long_force, velocity):
    """This function uses the rigid body strategy along with other topics including jacking to find the displacement
       of all four springs on the vehicle. The answers are returned in a 4x1 matrix: lf, rf, lr, RR.

        Rigid Body Theorem
        rr_disp = lr_disp + lf_disp -rf_disp

        Matrix Setup
        A Matrix
        |wheel rates|
        |moment about x axis|
        |moment about y axis|
        |rigid body theorem assuming chassis is rigid so all points on plane|

        x matrix
        |wheel displacement lf, wheel displacement rf, wheel displacement lr, wheel displacement RR|

        B matrix
        |sum z forces, sum moment x axis about cg, sum moment y_axis about cg, rigid body theorem|
        """
    # Decoupled Suspension
    # determine what corner springs are on then- sums them to get wheel rate for each corner
    # define wheel rates dictionaries for each corner
    # FR or RR(front roll / rear roll
    # FH or RH(front heave / rear heave)
    # LF, RF, LR, RR (common corner notation)
    lf_wr = OrderedDict([])
    rf_wr = OrderedDict([])
    rr_wr = OrderedDict([])
    lr_wr = OrderedDict([])
    # establishing wheel rates for de-coupled suspension
    for corner, k, mr in zip(dictionary['Performance Figures']['Spring Corner'],
                             dictionary['Performance Figures']['Spring Rate'],
                             dictionary['Performance Figures']['Motion Ratio']):
        if corner == "LF" or corner == "Front_Heave" or corner == "Front_Roll":
            lf_wr[corner] = k * mr**2
        if corner == "RF" or corner == "Front_Heave" or corner == "Front_Roll":
            rf_wr[corner] = k * mr ** 2
        if corner == "LR" or corner == "Rear_Heave" or corner == "Rear_Roll":
            lr_wr[corner] = k * mr ** 2
        if corner == "RR" or corner == "Rear_Heave" or corner == "Rear_Roll":
            rr_wr[corner] = k * mr ** 2
        else:
            pass

    # converting tire force lists to individual variables

    tire_force_lat_lf, tire_force_lat_rf, tire_force_lat_lr, tire_force_lat_rr = lat_force
    tire_force_long_lf, tire_force_long_rf, tire_force_long_lr, tire_force_long_rr = long_force

    mass = dictionary['Performance Figures']['Weight'][0] / 386.066
    lat_accel = (tire_force_lat_lf + tire_force_lat_rf + tire_force_lat_lr + tire_force_lat_rr) / mass
    long_accel = (tire_force_long_lf + tire_force_long_rf + tire_force_long_lr + tire_force_long_rr) / mass

    # Define the distance from wheel center to cg using suspension point dictionary
    lf_wc_to_cg_dist_y = dictionary['Left Front']['Wheel Center'][1] - \
        dictionary['Performance Figures']['Center of Gravity'][1]
    rf_wc_to_cg_dist_y = dictionary['Right Front']['Wheel Center'][1] - \
        dictionary['Performance Figures']['Center of Gravity'][1]
    lr_wc_to_cg_dist_y = dictionary['Left Rear']['Wheel Center'][1] - \
        dictionary['Performance Figures']['Center of Gravity'][1]
    rr_wc_to_cg_dist_y = dictionary['Right Rear']['Wheel Center'][1] - \
        dictionary['Performance Figures']['Center of Gravity'][1]
    lf_wc_to_cg_dist_x = dictionary['Left Front']['Wheel Center'][0] - \
        dictionary['Performance Figures']['Center of Gravity'][0]
    rf_wc_to_cg_dist_x = dictionary['Right Front']['Wheel Center'][0] - \
        dictionary['Performance Figures']['Center of Gravity'][0]
    lr_wc_to_cg_dist_x = dictionary['Left Rear']['Wheel Center'][0] - \
        dictionary['Performance Figures']['Center of Gravity'][0]
    rr_wc_to_cg_dist_x = dictionary['Right Rear']['Wheel Center'][0] - \
        dictionary['Performance Figures']['Center of Gravity'][0]

    side_jacking_list = jacking_calculations(dictionary, 'Side')
    front_jacking_list = jacking_calculations(dictionary, 'Front')

    # Aero Forces
    density_air = dictionary['Performance Figures']['Air Density'][0]
    frontal_area = dictionary['Performance Figures']['Frontal Area'][0]
    coeff_lift = dictionary['Performance Figures']['Coeff Lift'][0]
    aero_load = .5 * density_air * frontal_area * velocity ** 2

    # 4x4 matrix with 4th row being justified by rigid body table concept from "" book

    a_mat = np.array([[lf_wr['Front_Heave'] + lf_wr['LF'], rf_wr['Front_Heave'] + rf_wr['RF'],
                       lr_wr['Rear_Heave'] + lr_wr['LR'], rr_wr['Rear_Heave'] + rr_wr['RR']],
                      [(lf_wr['Front_Roll'] + lf_wr['LF']) * lf_wc_to_cg_dist_y,
                       (rf_wr['Front_Roll'] + rf_wr['RF']) * rf_wc_to_cg_dist_y,
                       (lr_wr['Rear_Roll'] + lr_wr['LR']) * lr_wc_to_cg_dist_y,
                       (rr_wr['Rear_Roll'] + rr_wr['RR']) * rr_wc_to_cg_dist_y],
                      [(lf_wr['Front_Heave'] + lf_wr['LF']) * lf_wc_to_cg_dist_x,
                       (rf_wr['Front_Heave'] + rf_wr['RF']) * rf_wc_to_cg_dist_x,
                       (lr_wr['Rear_Heave'] + lr_wr['LR']) * lr_wc_to_cg_dist_x,
                       (rr_wr['Rear_Heave'] + rr_wr['RR']) * rr_wc_to_cg_dist_x],
                      [-1 + dictionary['Performance Figures']['Shims'][0],
                       1 + dictionary['Performance Figures']['Shims'][1],
                       1 + dictionary['Performance Figures']['Shims'][2],
                       -1 + dictionary['Performance Figures']['Shims'][3]]])

    # 4x1 matrix

    b_mat = np.array([dictionary['Performance Figures']['Weight'][0] +
                      aero_load -
                      tire_force_lat_lf * front_jacking_list[0] -
                      tire_force_lat_rf * front_jacking_list[1] -
                      tire_force_lat_lr * front_jacking_list[2] -
                      tire_force_lat_rr * front_jacking_list[3] -
                      tire_force_long_lf * side_jacking_list[0] -
                      tire_force_long_rf * side_jacking_list[1] -
                      tire_force_long_lr * side_jacking_list[2] -
                      tire_force_long_rr * side_jacking_list[3],
                      -mass * dictionary['Performance Figures']['Center of Gravity'][2] * lat_accel -
                      tire_force_lat_lf * front_jacking_list[0] * lf_wc_to_cg_dist_y -
                      tire_force_lat_rf * front_jacking_list[1] * rf_wc_to_cg_dist_y -
                      tire_force_lat_lr * front_jacking_list[2] * lr_wc_to_cg_dist_y -
                      tire_force_lat_rr * front_jacking_list[3] * rr_wc_to_cg_dist_y -
                      tire_force_long_lf * side_jacking_list[0] * lf_wc_to_cg_dist_y -
                      tire_force_long_rf * side_jacking_list[1] * rf_wc_to_cg_dist_y -
                      tire_force_long_lr * side_jacking_list[2] * lr_wc_to_cg_dist_y -
                      tire_force_long_rr * side_jacking_list[3] * rr_wc_to_cg_dist_y,
                      mass * dictionary['Performance Figures']['Center of Gravity'][2] * long_accel -
                      tire_force_long_lf * side_jacking_list[0] * lf_wc_to_cg_dist_x -
                      tire_force_long_rf * side_jacking_list[1] * rf_wc_to_cg_dist_x -
                      tire_force_long_lr * side_jacking_list[2] * lr_wc_to_cg_dist_x -
                      tire_force_long_rr * side_jacking_list[3] * rr_wc_to_cg_dist_x -
                      tire_force_lat_lf * front_jacking_list[0] * lf_wc_to_cg_dist_x -
                      tire_force_lat_rf * front_jacking_list[1] * rf_wc_to_cg_dist_x -
                      tire_force_lat_lr * front_jacking_list[2] * lr_wc_to_cg_dist_x -
                      tire_force_lat_rr * front_jacking_list[3] * rr_wc_to_cg_dist_x,
                      0])

    # print 'front jacking list', front_jacking_list
    # print 'side jacking list', side_jacking_list
    # b_mat = np.array([dictionary['Performance Figures']['Weight'][0],
    #                   mass * dictionary['Performance Figures']['Center of Gravity'][2] * lat_accel,
    #                   mass * dictionary['Performance Figures']['Center of Gravity'][2] * long_accel,
    #                   0])
    # print b_mat[0]
    # print b_mat[1]
    # print a_mat, 'a '
    # print b_mat, 'b '
    # print dictionary['Performance Figures']['Weight']
    jacking_forces = [tire_force_lat_lf * front_jacking_list[0],
                      tire_force_lat_rf * front_jacking_list[1],
                      tire_force_lat_lr * front_jacking_list[2],
                      tire_force_lat_rr * front_jacking_list[3],
                      tire_force_long_lf * side_jacking_list[0],
                      tire_force_long_rf * side_jacking_list[1],
                      tire_force_long_lr * side_jacking_list[2],
                      tire_force_long_rr * side_jacking_list[3]]
    wheel_displacements = np.linalg.solve(a_mat, b_mat)
    # print wheel_displacements, "answer"
    return wheel_displacements, jacking_forces


def mode_displacements(wheel_disp_list):
    """
    CHANGE TO DICTIONARY INSTEAD OF LIST WHEN GIVEN OUT OF FOUR CORNER WHEEL DISP
    Wheel list in order lf, rf, lr, rr
    """


def load_transfer(acceleration, weight, base, cg_height):
    """Calculate load transfer in either the longitudinal or lateral direction when given either trackwidth or wheelbase
    INCORPORATE LLTD MOTHER TRUCKER
    """
    transfer = cg_height * weight * acceleration / base
    return transfer


def wheel_disp_compare_plot(dictionary, a_x, a_y, corner_weights_static, time, data_lf, data_rf, data_lr, data_rr,
                            velocity):
    """plots damper positions with given ax, and ay arrays"""
    # generate quarter elipse for all four tires
    weight = dictionary['Performance Figures']['Weight'][0]
    trackwidth_f = dictionary['Left Front']['Wheel Center'][1] - dictionary['Right Front']['Wheel Center'][1]
    trackwidth_r = dictionary['Left Front']['Wheel Center'][1] - dictionary['Right Front']['Wheel Center'][1]
    wheel_base = dictionary['Left Front']['Wheel Center'][0] - dictionary['Left Rear']['Wheel Center'][0]

    x_load_transfer = load_transfer(a_x, dictionary['Performance Figures']['Weight'][0], wheel_base,
                                    dictionary['Performance Figures']['Center of Gravity'][2])

    y_load_transfer = load_transfer(a_y, dictionary['Performance Figures']['Weight'][0], trackwidth_f,
                                    dictionary['Performance Figures']['Center of Gravity'][2])
    dynamics_loads = []
    accel_long = []
    accel_lat = []
    for x, y, a1, a2 in zip(x_load_transfer, y_load_transfer, a_x, a_y):
        lf = corner_weights_static[0] + x + y
        rf = corner_weights_static[1] + x - y
        lr = corner_weights_static[2] - x + y
        rr = corner_weights_static[3] - x - y
        # print 'dynamics loads', lf, rf, lr, rr
        dynamics_loads.append(np.array([lf, rf, lr, rr]))
        accel_long.append(np.array([a1, a1, a1, a1]))
        accel_lat.append(np.array([a2, a2, a2, a2]))
        # print dynamics_loads
    lat_tire_force = [load*a for load, a in zip(dynamics_loads, accel_lat)]
    long_tire_force = [load*a for load, a in zip(dynamics_loads, accel_long)]
    damper_displacements = []
    jacking_forces = []
    for ly, lx, v in zip(lat_tire_force, long_tire_force, velocity):
        # print 'lat_f', ly
        # print 'lat_x', lx
        displacements, jacking_f = four_corner_wheel_displacement(dictionary, ly, lx, v)
        damper_displacements.append(np.multiply(displacements, dictionary['Performance Figures']['Motion Ratio'][0:4]))
        jacking_forces.append(jacking_f)
    print jacking_forces

    ax = plt.subplot2grid((4, 2), (0, 0), projection='3d')
    ax2 = plt.subplot2grid((4, 2), (0, 1), projection='3d')
    ax3 = plt.subplot2grid((4, 2), (1, 0), projection='3d')
    ax4 = plt.subplot2grid((4, 2), (1, 1), projection='3d')
    ax5 = plt.subplot2grid((4, 2), (2, 0), projection='3d')
    ax6 = plt.subplot2grid((4, 2), (2, 1), projection='3d')
    ax7 = plt.subplot2grid((4, 2), (3, 0), colspan=2)

    ax.set_title('Damper Displacements')
    ax2.set_title('Normal Loads')
    ax3.set_title('Longitudinal Forces')
    ax4.set_title('Lateral Forces')
    ax5.set_title('Jacking Lateral')
    ax6.set_title('Jacking Longitudinal')
    ax7.set_title('Displacement vs Time')

    ax.plot(a_x, a_y, [x[0] for x in damper_displacements], c='r', label='lf')
    ax.plot(a_x, a_y, [x[1] for x in damper_displacements], c='orange', label='rf')
    ax.plot(a_x, a_y, [x[2] for x in damper_displacements], c='g', label='lr')
    ax.plot(a_x, a_y, [x[3] for x in damper_displacements], c='b', label='rr')

    ax2.plot(a_x, a_y, [x[0] for x in dynamics_loads], c='r', label='lf')
    ax2.plot(a_x, a_y, [x[1] for x in dynamics_loads], c='orange', label='rf')
    ax2.plot(a_x, a_y, [x[2] for x in dynamics_loads], c='g', label='lr')
    ax2.plot(a_x, a_y, [x[3] for x in dynamics_loads], c='b', label='rr')

    ax3.plot(a_x, a_y, [x[0] for x in long_tire_force], c='r', label='lf')
    ax3.plot(a_x, a_y, [x[1] for x in long_tire_force], c='orange', label='rf')
    ax3.plot(a_x, a_y, [x[2] for x in long_tire_force], c='g', label='lr')
    ax3.plot(a_x, a_y, [x[3] for x in long_tire_force], c='b', label='rr')

    ax4.plot(a_x, a_y, [x[0] for x in lat_tire_force], c='r', label='lf')
    ax4.plot(a_x, a_y, [x[1] for x in lat_tire_force], c='orange', label='rf')
    ax4.plot(a_x, a_y, [x[2] for x in lat_tire_force], c='g', label='lr')
    ax4.plot(a_x, a_y, [x[3] for x in lat_tire_force], c='b', label='rr')

    ax5.plot(a_x, a_y, [x[0] for x in jacking_forces], c='r', label='lf')
    ax5.plot(a_x, a_y, [x[1] for x in jacking_forces], c='orange', label='rf')
    ax5.plot(a_x, a_y, [x[2] for x in jacking_forces], c='g', label='lr')
    ax5.plot(a_x, a_y, [x[3] for x in jacking_forces], c='b', label='rr')

    ax6.plot(a_x, a_y, [x[4] for x in jacking_forces], c='r', label='lf')
    ax6.plot(a_x, a_y, [x[5] for x in jacking_forces], c='orange', label='rf')
    ax6.plot(a_x, a_y, [x[6] for x in jacking_forces], c='g', label='lr')
    ax6.plot(a_x, a_y, [x[7] for x in jacking_forces], c='b', label='rr')

    ax7.plot(time, [x[0] for x in damper_displacements], c='r', label='lf')
    ax7.plot(time, [x[1] for x in damper_displacements], c='orange', label='rf')
    ax7.plot(time, [x[2] for x in damper_displacements], c='g', label='lr')
    ax7.plot(time, [x[3] for x in damper_displacements], c='b', label='rr')
    ax7.plot(time, data_lf+.28, c='r', label='lf', linestyle=':')
    ax7.plot(time, data_rf+.28, c='orange', label='rf', linestyle=':')
    ax7.plot(time, data_lr+.35, c='g', label='lr', linestyle=':')
    ax7.plot(time, data_rr+.35, c='b', label='rr', linestyle=':')

    plt.legend()
    plt.show()
    return lat_tire_force, long_tire_force


def anti_squat_percent(ic_xz, wheel_center, wheel_base, center_gravity_height):
    """Takes longitudinal instant center, wheel base, wheel center, and cgh, and calculates the percentage """
    #  using angle finder from longitudinal instant center to wheel center.
    as_angle = (two_d_horizontal_angle([ic_xz[0], ic_xz[1]], [wheel_center[0], wheel_center[2]])) * math.pi / 180
    #  calculates anti-squat percent from tan(angle)*(WB/c of g height) * 100
    as_tan = math.tan(as_angle)
    as_percent = (as_tan / (center_gravity_height / wheel_base)) * 100
    return as_percent
