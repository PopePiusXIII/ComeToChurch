from __future__ import division
import sys
sys.path.insert(0, 'C:\Users\\harle\\Documents\\ComeToChurch\\Tire Code')
import Fy as Fy
import MRA_Norm as mra_norm
import matplotlib.pyplot as plt
import numpy as np
import math


def steering_angle(cg_to_front_wheels_len, slip_angle, velocity, beta, acceleration, track_width):
    steer_angle = 180 / math.pi * (cg_to_front_wheels_len / ((velocity**2/acceleration)+track_width/2) + beta
                                   + slip_angle)
    return steer_angle


def beta_angle(cg_to_rear_wheels_len, slip_angle, velocity, acceleration, track_width):
    beta = -180 / math.pi * (cg_to_rear_wheels_len/((velocity**2/acceleration)+track_width/2)) + slip_angle
    return beta


def moment_sum(cg_to_front_wheels_len, cg_to_rear_wheels_len, alpha_f, alpha_r, coeff, mue_lf, mue_rf, mue_rr, mue_lr,
               load_lf, load_rf, load_rr, load_lr, stiffness_lf, stiffness_rf, stiffness_rr, stiffness_lr, scale_factor):
    """anti clock wise positive moment
    clockwise from vertical is positive slip, beta, delta"""
    # plt.plot(alpha_f)
    # plt.plot(alpha_r)
    # plt.show()
    coeff_left = coeff
    coeff_right = coeff
    coeff_right[2] = -coeff_right[2]
    # slip angle alpha front axle  to right
    lf_tire_force = mra_norm.norm_expansion(coeff_left, mra_norm.norm_slip_angle(stiffness_lf, load_lf,
                                                                                 alpha_f, mue_lf), scale_factor,
                                            load_lf, mue_lf, stiffness_lf)

    rf_tire_force = -mra_norm.norm_expansion(coeff_right, mra_norm.norm_slip_angle(stiffness_rf, load_rf,
                                                                                   -alpha_f, mue_rf), scale_factor,
                                             load_rf, mue_rf, stiffness_rf)

    # moment_f = rf_tire_force * cg_to_front_wheels_len
    #
    moment_lf = -lf_tire_force * cg_to_front_wheels_len

    moment_rf = -rf_tire_force * cg_to_front_wheels_len

    # slip angle alpha rear axle to right hand turn
    lr_tire_force = mra_norm.norm_expansion(coeff_left, mra_norm.norm_slip_angle(stiffness_lr, load_lr,
                                                                                 alpha_r, mue_lr), scale_factor,
                                            load_lr, mue_lr, stiffness_lr)

    rr_tire_force = -mra_norm.norm_expansion(coeff_right, mra_norm.norm_slip_angle(stiffness_rr, load_rr,
                                                                                   -alpha_r, mue_rr), scale_factor,
                                             load_rr, mue_rr, stiffness_rr)
    # moment_r = rr_tire_force * cg_to_rear_wheels_len
    moment_lr = lr_tire_force * cg_to_rear_wheels_len

    moment_rr = rr_tire_force * cg_to_rear_wheels_len

    # drag moment
    # moment_rr = (-np.sin(abs(alpha_r) / 57.3) * rr_tire_force * half_trackwidth -
    #              np.cos(abs(alpha_r) / 57.3) * rr_tire_force * cg_to_rear_wheels_len)
    # plt.plot(moment_lf, c='orange')
    # plt.plot(moment_rf, c='r')
    # plt.plot(moment_lr, c='y')
    # plt.plot(moment_rr, c='g')
    # plt.plot(lf_tire_force, c='orange', linestyle='--')
    # plt.plot(rf_tire_force, c='r', linestyle=':')
    # plt.plot(rear_tire_force, c='y', linestyle='--')
    # plt.plot(rear_tire_force, c='g', linestyle=':')
    # print 'moments', moment_f, moment_r
    # print 'DIFFERENCE', moment_lf - moment_rf
    # print'LF', moment_lf, '\n RF', moment_rf, 'RR', moment_rr, '\n LR', moment_lr
    # sum_moment = moment_lf + moment_rf + moment_rr + moment_lr
    print 'forces', lf_tire_force[0], rf_tire_force[0], lr_tire_force[0], rr_tire_force[0]
    print 'momentst', moment_lf[0], moment_rf[0], moment_lr[0], moment_rr[0]
    sum_moment = moment_lf + moment_rf + moment_lr + moment_rr
    return sum_moment


def lat_accel(alpha_f, alpha_r, coeff, car_weight, mue_lf, mue_rf, mue_rr, mue_lr, load_lf, load_rf, load_rr, load_lr,
              stiffness_lf, stiffness_rf, stiffness_rr, stiffness_lr, scale_factor):
    """For the given parameters a lateral acceleration will be returned"""
    coeff_left = coeff
    coeff_right = coeff
    coeff_right[2] = -coeff_right[2]

    lat_lf = mra_norm.norm_expansion(coeff_left, mra_norm.norm_slip_angle(stiffness_lf, load_lf, alpha_f, mue_lf),
                                     scale_factor, load_lf, mue_lf, stiffness_lf)
    lat_rf = -mra_norm.norm_expansion(coeff_right, mra_norm.norm_slip_angle(stiffness_rf, load_rf, -alpha_f, mue_rf),
                                      scale_factor, load_rf, mue_rf, stiffness_rf)
    lat_lr = mra_norm.norm_expansion(coeff_left, mra_norm.norm_slip_angle(stiffness_lr, load_lr, alpha_r, mue_lr),
                                     scale_factor, load_lr, mue_lr, stiffness_lr)
    lat_rr = -mra_norm.norm_expansion(coeff_right, mra_norm.norm_slip_angle(stiffness_rr, load_rr, -alpha_r, mue_rr),
                                      scale_factor, load_rr, mue_rr, stiffness_rr)
    tot_lat_force = lat_lf + lat_lr + lat_rf + lat_rr
    # print 'lateral force', tot_lat_force
    # print 'lats', lat_f, lat_r
    lateral_accel = tot_lat_force / car_weight
    # print lateral_accel
    return lateral_accel


def weight_transfer(accel, weight, cg_height, track_width):
    """Returns the amount of weight transferred from one side or axle to the other from inertial forces
    left hand turn positive accel"""
    transfer = accel * weight * cg_height / track_width
    return transfer


def mmm_diagram(max_slip_f, max_slip_r, velocity, cg_to_front_wheels_len, cg_to_rear_wheels_len,
                load_lf, load_rf, load_rr, load_lr, stiffness_lf, stiffness_rf, stiffness_rr, stiffness_lr, mue_lf,
                mue_rf, mue_rr, mue_lr, coeff, car_weight, half_trackwidth, scale_factor):
    alpha_f = []
    alpha_r = []
    # color = ['r', 'o', 'y', 'g', 'b', 'c', 'k']
    # label = []
    for i in range(-max_slip_f, max_slip_f, 1):
        for j in range(-max_slip_r, max_slip_r, 1):

            alpha_f.append(i)
            alpha_r.append(j)

    alpha_f = np.array(alpha_f)
    alpha_r = np.array(alpha_r)

    # calculate mass acceleration vs yaw acceleration using other functions
    # initial guess w/o weight transfer
    accel = [lat_accel(alpha_f, alpha_r, coeff, car_weight, mue_lf, mue_rf, mue_rr, mue_lr,
                      load_lf, load_rf, load_rr, load_lr, stiffness_lf, stiffness_rf, stiffness_rr, stiffness_lr,
                       scale_factor)]
    moment = [moment_sum(cg_to_front_wheels_len, cg_to_rear_wheels_len, alpha_f, alpha_r, coeff, mue_lf, mue_rf, mue_rr,
                        mue_lr, load_lf, load_rf, load_rr, load_lr, stiffness_lf, stiffness_rf, stiffness_rr,
                        stiffness_lr, scale_factor)]
    error = []
    weight_error = []
    for i in range(1, 100, 1):
        load_transfer = weight_transfer(accel[i-1], weight, 11, 47)
        dyn_load_lf, dyn_load_rf, dyn_load_rr, dyn_load_lr = (load_lf - load_transfer/2, load_rf + load_transfer/2,
                                                              load_rr + load_transfer/2, load_lr - load_transfer/2)
        dyn_load_lf[dyn_load_lf < 0] = 0
        dyn_load_rf[dyn_load_rf < 0] = 0
        dyn_load_rr[dyn_load_rr < 0] = 0
        dyn_load_lr[dyn_load_lr < 0] = 0

        accel.append(lat_accel(alpha_f, alpha_r, coeff, car_weight, mue_lf, mue_rf, mue_rr, mue_lr,
                               dyn_load_lf, dyn_load_rf, dyn_load_rr, dyn_load_lr, stiffness_lf, stiffness_rf,
                               stiffness_rr, stiffness_lr, scale_factor))
        moment.append(moment_sum(cg_to_front_wheels_len, cg_to_rear_wheels_len, alpha_f, alpha_r, coeff, mue_lf, mue_rf,
                                 mue_rr, mue_lr, dyn_load_lf, dyn_load_rf, dyn_load_rr, dyn_load_lr, stiffness_lf,
                                 stiffness_rf, stiffness_rr, stiffness_lr, scale_factor))

        error.append(np.sum(np.subtract(accel[i], accel[i - 1])) ** 2)
        weight_error.append(dyn_load_lf + dyn_load_rf + dyn_load_rr + dyn_load_lr)

        if error[-1] < 1e-6:
            plt.plot(weight_error)
            plt.show()
            accel = accel[i]
            moment = moment[i]
            print 'ERROR', error
            plt.plot(error)
            plt.show()
            print i
            break
        elif i == 99:
            error.append(np.sum(np.subtract(accel[i], accel[i-1]))**2)
            plt.plot(error[0:])
            plt.show()
            print "WTFFFFFFFFFFF"

    # sort result into beta iso lines and steering iso lines (aka use beta or steer one at a time as an open variable)
    # steering iso lines
    # splice in groups of length beta data to hold steering constant varying beta
    length_alpha_r = max_slip_r * 2
    length_alpha_f = max_slip_f * 2

    for i in range(0, length_alpha_f, 1):
        steering_x = accel[i*length_alpha_r:length_alpha_r*i+length_alpha_r]
        steering_y = moment[i*length_alpha_r:length_alpha_r*i+length_alpha_r]
        if i != length_alpha_f / 2:
            plt.plot(steering_x, steering_y, color='g')
        else:
            plt.plot(steering_x, steering_y, color='k')
    # beta iso lines
    # skip items to hold beta constant but vary steering
    beta_x = []
    beta_y = []
    for i in range(0, length_alpha_f, 1):
        for j in range(0, length_alpha_r, 1):
            beta_x.append(accel[i + j * length_alpha_f])
            beta_y.append(moment[i + j * length_alpha_f])
            if i != length_alpha_f / 2:
                plt.plot(beta_x, beta_y, color='b')
            else:
                plt.plot(beta_x, beta_y, color='r')

        beta_x = []
        beta_y = []
    plt.grid(True)
    plt.show()
    return

max_slip_f = 12
max_slip_r = 12
turn_radiuss = 100
front_to_cgg = 2.5
rear_to_cgg = 2.5
load_lf = 130
load_lr = 130
load_rf = 130
load_rr = 130
weight = load_lf + load_lr + load_rf + load_rr

data = Fy.fy_data_collector("C:\Users\harle\Documents\ComeToChurch\Raw Data\Round 6\RawData_13inch_Cornering_ASCII_USCS\A1654raw12.dat")
coefff, mue_sensitivity, stiffness_sensitivity,  = mra_norm.non_dim_slip_angle_force_fit(data["Slip Angle (Steady)"]['25']['12.0']['0'],
                                                                                         data['Fy (Steady)']['25']['12.0']['0'],
                                                                                         data['Fz (Steady)']['25']['12.0']['0'])
stiffness_lf = mra_norm.mue_function(stiffness_sensitivity, load_lf)
stiffness_rf = mra_norm.mue_function(stiffness_sensitivity, load_rf)
stiffness_rr = mra_norm.mue_function(stiffness_sensitivity, load_rr)
stiffness_lr = mra_norm.mue_function(stiffness_sensitivity, load_lr)
mue_lf = mra_norm.mue_function(mue_sensitivity, load_lf)
mue_rf = mra_norm.mue_function(mue_sensitivity, load_rf)
mue_rr = mra_norm.mue_function(mue_sensitivity, load_rr)
mue_lr = mra_norm.mue_function(mue_sensitivity, load_lr)
mmm_diagram(max_slip_f, max_slip_r, turn_radiuss, front_to_cgg, rear_to_cgg, load_lf, load_rf, load_rr, load_lr, stiffness_lf,
            stiffness_rf, stiffness_rr, stiffness_lr, mue_lf, mue_rf, mue_rr, mue_lr, coefff, weight, 2, .66)

