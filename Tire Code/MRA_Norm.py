from __future__ import division
import Tire_Fitting as Tfit
import Fy as Fy
import numpy as np
import matplotlib.pyplot as plt


def mue_finder(fz_list, force_list):
    """Returns mue at the max force value of either a negative or positive force list
    MUE: is defined in EXTENSION OF THE NON-DIMENSIONAL TIRE THEORY TO GENERAL OPERATING CONDITION BY: ED K
    MUE = MAX(FX) / CORRESPONDING LOAD"""
    max_force = (max(min(force_list), max(force_list), key=abs))
    mue = max_force / fz_list[force_list.index(max_force)]
    return abs(mue)


def norm_force(fz_val, force_array, mue_val):
    """ returns list of normalized fx data between 0 and 1
    REFER: page 479 RCVD
    EQUATION: Fx / (MUE * Fz)
    NOTE: ONLY NORMALIZES FOR ONE PAIR OF LIST/ARRAYS
    EXAMPLE: slip_list[0][0][0][0][0], force_list[0][0][0][0][0]"""

    normal_force = (force_array / (mue_val * fz_val))

    return normal_force


def norm_slip_ratio(k_x, fz_list, slip_list, force_list):
    """returns a list of normalized slip_ratios
    REFER: page 479 RCVD
    EQUATION: (K_x * SLIP_RATIO)/ (MUE * Fz)
    k_x: longitudinal stiffness
    """
    norm_sr_list = []
    mue = mue_finder(fz_list, force_list)
    pair = zip(fz_list, slip_list)  # create list of (fz, force) tuple
    for values in pair:
        norm_sr_list.append((k_x * values[1]) / (mue * values[0]))

    return norm_sr_list


def norm_slip_angle(c_y, fz_val, slip_array, mue):
    """returns a list of normalized slip_angles
    REFER: page 478 RCVD
    EQUATION: (C * SLIP ANGLE)/ (MUE * Fz)
    slip: degree
    c_y: cornering stiffness lbf / deg
    NOTICE: DIMENSIONLESS
    """
    # print'cy %s, slip %s, mue %s, fz_val %s' % (c_y, fz_val, slip_array, mue)
    norm_sa = ((c_y * slip_array) / (mue * fz_val))
    return norm_sa


def mue_function(coeff, load):
    mue = coeff[0] * load ** 3 + coeff[1] * load ** 2 + coeff[2] * load + coeff[3]
    return mue


def raw_norm_slip_force_plot(norm_slip_list, norm_force_list, color, label):
    """Scatter plot slip vs force
     User can pass color and label info as well"""
    if color is None:
        color = 'b'
    if label is None:
        label = None
    plt.scatter(norm_slip_list, norm_force_list, facecolor='none', edgecolors=color, label=label)
    plt.show()
    return None


def non_dim_slip_angle_force_fit(both_slip_sign_slip_list, both_slip_sign_force_list, fz_list):
    """for the list give the dimension that encompasses both positive and negative slip list
    Example of parameter: both_slip_sign_slip_list = data[speed][pressure][camber]
    returns a non dimensionlized fit model"""

    normal_slip = []
    normal_force = []
    mue = []
    load = []
    stiffness = []
    tested_loads = [50, 100, 150, 200, 250, 350]
    tested_load_str = ['-50', '-100', '-150', '-200', '-250', '-350']
    # colors = ['k', 'b', 'g', 'y', 'orange', 'r']
    for i in both_slip_sign_slip_list.keys():  # i = [load][data]  # positive and negative slip signs[-, +]
        for j in both_slip_sign_slip_list[i].keys():    # retieves loads in dict for each slip sign
            load_specific_slip = both_slip_sign_slip_list[i][j]
            load_specific_force = both_slip_sign_force_list[i][j]
            load_specific_fz = fz_list[i][j]

            if len(load_specific_slip) > 50:
                print 'list longer than 50'

                # calculate the stiffness and coefficient of friction each load. Both require list so np.arrays are nono
                stiffness.append(Fy.cornering_stiffness_calc(load_specific_slip, load_specific_force))
                mue.append(mue_finder(load_specific_fz, load_specific_force))
                load.append(-float(j))
                if stiffness[-1] == 0:
                    del stiffness[-1]
                    del mue[-1]
                    del load[-1]
                    print 'stiffness = 0'
                    continue

                # calculate norm slip
                normal_slip.append(norm_slip_angle(stiffness[-1], np.array(load_specific_fz),
                                                   np.array(load_specific_slip), mue[-1]))
                # calculate norm force
                normal_force.append(norm_force(np.array(load_specific_fz), np.array(load_specific_force), mue[-1]))
                # plt.scatter(normal_slip[-1], normal_force[-1], facecolor='none', edgecolors=colors[j])

    # Optional graph to visualize load sensitivity
    mue_sensitivity_coefficients = np.polyfit(load, mue, 3)
    stiffness_sensitivity_coefficients = np.polyfit(load, stiffness, 3)
    plt.plot(np.linspace(0, 350, 50), .6 * mue_function(mue_sensitivity_coefficients, np.linspace(0, 350, 50)))
    plt.plot(np.linspace(0, 350, 50), .6 * mue_function(stiffness_sensitivity_coefficients, np.linspace(0, 350, 50)))
    # plt.scatter(load, stiffness)
    # plt.scatter(load, .mue)
    plt.show()
    # flatten or the data so all of it can be fit with one magic formula curve
    flat_normal_slip = [item for sublist in normal_slip for item in sublist]
    flat_normal_force = [item for sublist in normal_force for item in sublist]
    coefficients = Tfit.least_square(flat_normal_slip, flat_normal_force)
    return coefficients, mue_sensitivity_coefficients, stiffness_sensitivity_coefficients


def norm_expansion(coefficients, norm_slip_val, scaling_factor, load, mue, stiffness):
    """expand the normalized mra data back out for force predictions at specified loads
    coefficients = [B, C, D, E, Sv, Sh] generated from (non_dim_slip_angle_force_fit)
    FZ = [load_1, load_2, load_n] lbf
    mue = [mue_1, mue_2, mue_n] lbf / lbf
    stiffness = [stiff_1, stiff_2, stiff_n] lbf/deg

    THEORY: Slip input must be normalized to input into the norm pacejka equation """

    # print'MUE   %s \n load %s \n   stiffness %s' % (mue, load, stiffness)
    # print 'norm slip', norm_slip_val

    expanded_force = (Tfit.model(norm_slip_val, coefficients) * load * mue * scaling_factor)
    return expanded_force
