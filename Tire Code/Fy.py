from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy


def fy_data_collector(file_path):
    """import .dat file from calspan and return raw and filtered/sorted data arrays of shape
        filtered[speed][pressure][camber][slip_sign][load][data_point]
        speed[25, 45]
        slip_sign[-0, +1]
        pressure[8, 10, 12, 14]
        camber[0, 1, 2, 3, 4]
        load[50, 100, 150, 200, 250, 300, 350]"""
    if file_path is None:
        return
    # opening the dat file to be sorted
    first_row = 4
    print file_path
    file_location = file_path
    # skip first 4 rows to jump strings    unpack=(True transposes the matrix)    dtype=float_ to turn it into float
    raw_data = np.loadtxt(file_location, skiprows=first_row, unpack=True, dtype='float_')
    data_dict = OrderedDict([
        ('Time', raw_data[0]),
        ('Speed', raw_data[1]),
        ('Slip Angle', raw_data[3]),
        ('Camber', raw_data[4]),
        ('Pressure', raw_data[7]),
        ('Fx', raw_data[8]),
        ('Fy', raw_data[9]),
        ('Fz', raw_data[10]),
        ('Mx', raw_data[11]),
        ('Mz', raw_data[12]),
        ('tsti', raw_data[16]),
        ('tstc', raw_data[17]),
        ('tsto', raw_data[18]),
        ('Slip Ratio', raw_data[20]),
        ('Sampling Frequency', 100)
    ])

    # slip velocity to filter out transient data and to target a tire building grip
    # multiply by frequency to put the units into deg / sec
    data_dict['Slip Velocity'] = np.diff(data_dict['Slip Angle'] * data_dict['Sampling Frequency'])
    # add one data point at the end to preserve the length
    data_dict['Slip Velocity'] = np.append(data_dict['Slip Velocity'], [1])

    # INDEX THE LOADS, CAMBER, SPEED, PRESSURE
    possible_loads = [-50.0, -100.0, -150, -200, -250, -350]
    possible_pressures = [8.0, 10.0, 12.0, 14.0]
    possible_cambers = [0, 1, 2, 3, 4]
    possible_speeds = [25, 45]
    data_length = 0
    possible_slip = ['Negative', 'Positive']
    pressure_key, camber_key, load_key, speed_key, slip_key = [None] * 5

    # BUILD GENERIC EMPTY LIST STRUCTURE
    # example list[speed][pressure][camber][slip_sign][load][data_point]

    filtered_fy = OrderedDict([(str(speed_key),
                                OrderedDict([(str(pressure_key),
                                              OrderedDict([(str(camber_key),
                                                            OrderedDict([(str(slip_key),
                                                                        OrderedDict([(str(load_key),
                                                                                    [])
                                                                                     for load_key in possible_loads]))
                                                                         for slip_key in ['Negative', 'Positive']]))
                                                           for camber_key in possible_cambers]))
                                             for pressure_key in possible_pressures]))
                               for speed_key in possible_speeds])
    filtered_sa = copy.deepcopy(filtered_fy)
    filtered_fx = copy.deepcopy(filtered_fy)
    filtered_sr = copy.deepcopy(filtered_fy)
    filtered_mx = copy.deepcopy(filtered_fy)
    filtered_mz = copy.deepcopy(filtered_fy)
    filtered_fz = copy.deepcopy(filtered_fy)

    # outer most for loop will loop through the entire length of dat file
    for i in range(len(data_dict['Slip Angle'])):
        # check to see if fz in range
        for load in possible_loads:
            if 1.1 * load < data_dict['Fz'][i] < .95 * load:   # watch signs holy fuck
                # load_index = possible_loads.index(j)  # will use this index to place data in final multi d array
                load_key = str(load)  # will use this index to place data in final multi d array
                break
            elif load == possible_loads[-1]:   # only if the value fails all cases wll this be true because break
                load_key = None

        # check to see if camber in range
        for camber in possible_cambers:
            if camber - .5 < data_dict['Camber'][i] < .5 + camber:
                # camber_index = possible_cambers.index(j)    # will use this index to place data in final multi d array
                camber_key = str(camber)   # will use this index to place data in final multi d array
                break
            elif camber == possible_cambers[-1]:     # only if the value fails all cases wll this be true because break
                camber_key = None

        # check to see if pressure in range
        for pressure in possible_pressures:
            if pressure - .5 < data_dict['Pressure'][i] < pressure + .5:
                # pressure_index = possible_pressures.index(j)    # use this index to place data in final multi d array
                pressure_key = str(pressure)    # use this index to place data in final multi d array
                break
            elif pressure == possible_pressures[-1]:   # only if the value fails cases wll this be true because break
                pressure_key = None

        #   check to see if speed in range
        for speed in possible_speeds:
            if speed - 1 < data_dict['Speed'][i] < speed + 1:
                # speed_index = possible_speeds.index(j)  # use this index to place data in final multi d array
                speed_key = str(speed)  # use this index to place data in final multi d array
                break
            elif speed == possible_speeds[-1]:  # only if the value fails cases wll this be true because break
                speed_key = None

        # check for pure fx by excluding data with slip angles above absolute value of one
        for j in range(2):
            if (-10 < data_dict['Slip Angle'][i] < 0) and (3.7 < data_dict['Slip Velocity'][i] < 4.3):
                # slip_index = 0  # only used for pass fail
                slip_key = 'Negative'  # only used for pass fail
                break
            elif (0 < data_dict['Slip Angle'][i] < 10) and (3.7 < data_dict['Slip Velocity'][i] < 4.3):
                # slip_index = 1
                slip_key = 'Positive'
            else:
                slip_key = None

        if pressure_key is not None and load_key is not None and camber_key is not None and speed_key is not \
                None and slip_key is not None:
            data_length += 1
            filtered_fy[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Fy'][i])
            filtered_mx[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Mx'][i])
            filtered_mz[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Mz'][i])
            filtered_sa[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Slip Angle'][i])
            filtered_fz[speed_key][pressure_key][camber_key][slip_key][load_key].append(-data_dict['Fz'][i])
            filtered_fx[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Fx'][i])
            filtered_sr[speed_key][pressure_key][camber_key][slip_key][load_key].append(data_dict['Slip Ratio'][i])

    # create dictionary to avoid confusion when calling channels
    result_dict = {'Time (Raw)': data_dict['Time'],
                   'Slip_Angle (Raw)': data_dict['Slip Angle'],
                   'Camber (Raw)': data_dict['Camber'],
                   'Pressure (Raw)': data_dict['Pressure'],
                   'Fx (Raw)': data_dict['Fx'],
                   'Fy (Raw)': data_dict['Fy'],
                   'Fz (Raw)': data_dict['Fz'],
                   'Mx (Raw)': data_dict['Mx'],
                   'Mz (Raw)': data_dict['Mz'],
                   'Slip Angle (Steady)': filtered_sa,
                   'Fy (Steady)': filtered_fy,
                   'Mx (Steady)': filtered_mx,
                   'Fz (Steady)': filtered_fz,
                   'Mz (Steady)': filtered_mz,
                   'Slip Ratio (Steady)': filtered_mz,
                   'Speed (Raw)': speed,
                   'tsti (Raw)': data_dict['tsti'],
                   'tsto (Raw)': data_dict['tsto'],
                   'tstc (Raw)': data_dict['tstc'],
                   'Slip Ratio (Raw)': data_dict['Slip Ratio'],
                   'Slip Sign (Test)': possible_slip,
                   'Speed (Test)': possible_speeds,
                   'Pressure (Test)': possible_pressures,
                   'Camber (Test)': possible_cambers,
                   'Load (Test)': possible_loads,
                   }

    return result_dict


def fx_data_collector(file_path):
    """import .dat file from calspan and return raw and filtered/sorted data arrays of shape
    filtered[speed][slip_sign][pressure][camber][load][data_point]
    speed[25, 45]
    slip_sign[-0, +1]
    pressure[8, 10, 12, 14]
    camber[0, 1, 2, 3, 4]
    load[50, 100, 150, 200, 250, 300, 350]"""

    # opening the dat file to be sorted
    first_row = 4
    file_location = file_path
    # skip first 4 rows to jump strings    unpack=(True transposes the matrix)    dtype=float_ to turn it into float
    raw_data = np.loadtxt(file_location, skiprows=first_row, unpack=True, dtype='float_')
    data_dict = OrderedDict([
        ('Time', raw_data[0]),
        ('Speed', raw_data[1]),
        ('Slip Angle', raw_data[3]),
        ('Camber', raw_data[4]),
        ('Pressure', raw_data[7]),
        ('Fx', raw_data[8]),
        ('Fy', raw_data[9]),
        ('Fz', raw_data[10]),
        ('Mx', raw_data[11]),
        ('Mz', raw_data[12]),
        ('tsti', raw_data[16]),
        ('tstc', raw_data[17]),
        ('tsto', raw_data[18]),
        ('Slip Ratio', raw_data[20]),
        ('Sampling Frequency', 100)
    ])

    # INDEX THE LOADS, CAMBER, SPEED, PRESSURE
    possible_loads = [-50.0, -100.0, -150, -200, -250, -350]
    possible_pressures = [8.0, 10.0, 12.0, 14.0]
    possible_cambers = [0, 1, 2, 3, 4]
    possible_speeds = [25, 45]
    possible_slip = [0]
    data_length = 0
    pressure_key, camber_key, load_key, speed_key, slip_key, slip_ratio_key = [None] * 6

    # BUILD GENERIC EMPTY LIST STRUCTURE
    # example list[speed][sr_sign][pressure][camber][load][data_point]
    filtered_fy = OrderedDict([(speed_key,
                                OrderedDict([(pressure_key,
                                              OrderedDict([(camber_key,
                                                            OrderedDict([(slip_key,
                                                                          OrderedDict([(load_key,
                                                                                        []) for load_key in
                                                                                      possible_loads]))
                                                                         for slip_key in ['Negative', 'Positive']]))
                                                           for camber_key in possible_cambers]))
                                             for pressure_key in possible_pressures]))
                               for speed_key in possible_speeds])
    filtered_fx = copy.deepcopy(filtered_fy)
    filtered_sr = copy.deepcopy(filtered_fy)
    filtered_mx = copy.deepcopy(filtered_fy)
    filtered_fz = copy.deepcopy(filtered_fy)

    # outer most for loop will loop through the entire length of dat file
    for i in range(len(data_dict['Slip Ratio'])):

        # check to see if fz in range
        for load in possible_loads:
            if 1.1 * load < data_dict['Fz'][i] < .95 * load:  # watch signs holy fuck
                load_key = load  # will use this index to place data in final multi d array
                break
            elif load == possible_loads[-1]:  # only if the value fails all cases wll this be true because break
                load_key = None

        # check to see if camber in range
        for camber in possible_cambers:
            if camber - .5 < data_dict['Camber'][i] < .5 + camber:
                camber_key = camber  # will use this index to place data in final multi d array
                break
            elif camber == possible_cambers[-1]:  # only if the value fails all cases wll this be true because break
                camber_key = None

        # check to see if pressure in range
        for pressure in possible_pressures:
            if pressure - .5 < data_dict['Pressure'][i] < pressure + .5:
                pressure_key = pressure  # use this index to place data in final multi d array
                break
            elif pressure == possible_pressures[-1]:  # only if the value fails cases wll this be true because break
                pressure_key = None

        # check to see if speed in range
        for speed in possible_speeds:
            if speed - 1 < data_dict['Speed'][i] < speed + 1:
                speed_key = speed   # use this index to place data in final multi d array
                break
            elif speed == possible_speeds[-1]:  # only if the value fails cases wll this be true because break
                speed_key = None

        # check for pure fx by excluding data with slip angles above absolute value of one
        for slip in possible_slip:
            if -1 < data_dict['Slip Angle'][i] < 1:
                slip_key = slip  # only used for pass fail
                break
            elif slip == possible_slip[-1]:  # only if the value fails cases wll this be true because break
                slip_key = None

        # sort data based off slip ratio sign. Need this to be able to fit separate lines to each side
        if data_dict['Slip Ratio'][i] < 0:
            slip_ratio_key = 'Negative'  # '0' represents the negative slip ratio index
        elif data_dict['Slip Ratio'][i] > 0:  # only if the value fails cases wll this be true because break
            slip_ratio_key = 'Positive'  # '1' represents the positive slip ratio index

        if pressure_key is not None and load_key is not None and camber_key is not None and speed_key is not \
                None and slip_key is not None:
            data_length += 1
            filtered_fx[speed_key][pressure_key][camber_key][slip_ratio_key][load_key].append(data_dict['Fx'][i])
            filtered_mx[speed_key][pressure_key][camber_key][slip_ratio_key][load_key].append(data_dict['Mx'][i])
            filtered_sr[speed_key][pressure_key][camber_key][slip_ratio_key][load_key].append(data_dict['Slip Ratio'][i])
            filtered_fz[speed_key][pressure_key][camber_key][slip_ratio_key][load_key].append(-data_dict['Fz'][i])

    result_dict = {'Time (Raw)': data_dict['Time'],
                   'Slip_Angle (Raw)': data_dict['Slip Angle'],
                   'Camber (Raw)': data_dict['Camber'],
                   'Pressure (Raw)': data_dict['Pressure'],
                   'Fx (Raw)': data_dict['Fx'],
                   'Fy (Raw)': data_dict['Fy'],
                   'Fz (Raw)': data_dict['Fz'],
                   'Mx (Raw)': data_dict['Mx'],
                   'Mz (Raw)': data_dict['Mz'],
                   'Slip Ratio (Steady)': filtered_sr,
                   'Fy (Steady)': filtered_fy,
                   'Fx (Steady)': filtered_fx,
                   'Mx (Steady)': filtered_mx,
                   'Fz (Steady)': filtered_fz,
                   'Speed (Raw)': speed,
                   'tsti (Raw)': data_dict['tsti'],
                   'tsto (Raw)': data_dict['tsto'],
                   'tstc (Raw)': data_dict['tstc'],
                   'Slip Sign (Test)': possible_slip,
                   'Speed (Test)': possible_speeds,
                   'Pressure (Test)': possible_pressures,
                   'Camber (Test)': possible_cambers,
                   'Load (Test)': possible_loads,
                   'Slip Ratio (Test)': ['Negative', 'Positive'],
                   }
    return result_dict


def cornering_stiffness_calc(slip_list, force_list):
    """calculate the cornering of the given (slip_angle, fy)
    NOTE: ONLY CALCULATES THE CORNERING FOR ONE PAIR OF LIST/ARRAYS
    EXAMPLE: slip_list[0][0][0][0][0], force_list[0][0][0][0][0]"""
    # only select data close to the origin
    small_slip = []
    small_force = []
    linear_limit = int(abs(max(min(force_list), max(force_list), key=abs)) * .4)
    for force in force_list:
        if -linear_limit < force < linear_limit:   # only take values between -1 and 1
            small_slip.append(slip_list[force_list.index(force)])
            small_force.append(force)

    if 3 < len(small_slip):  # check to make sure small_slip was populated before fitting
        cornering_stiffness = (np.polyfit(small_slip, small_force, 1))
        return abs(cornering_stiffness[0])
    else:
        print'list poorly constrained, less than 10 data points'
        return 0


def longitudinal_stiffness_calc(slip_list, force_list):
    """calculate the longitudinal stiffness of the given (slip_ratio, fx)
    NOTE: ONLY CALCULATES THE STIFFNESS FOR ONE PAIR OF LIST/ARRAYS, NEEDS POSITVE(+) AND NEGATIVE(-) SLIP CONDITIONS
    EXAMPLE: slip_list[0][0][0][0][0], force_list[0][0][0][0][0]"""
    # only select data close to the origin
    small_slip = []
    small_force = []
    linear_limit = int(abs(max(min(force_list), max(force_list), key=abs))*.4)
    print linear_limit
    for force in force_list:
        if -linear_limit < force < linear_limit:   # only take values between -1 and 1
            small_slip.append(slip_list[force_list.index(force)])
            small_force.append(force)

    if 5 < len(small_slip):  # check to make sure small_slip was populated before fitting
        longitudinal_stiffness = (np.polyfit(small_slip, small_force, 1))
        print len(small_slip), len(small_force), 'SUCCCCESSSS'
        return abs(longitudinal_stiffness[0])
    else:
        print'list poorly constrained, less than 10 data points'
        return [0, 0]


def all_channels_vs_time(data, color):
    fig = plt.figure(611)
    ax = fig.add_subplot(611)
    ax1 = fig.add_subplot(612)
    ax2 = fig.add_subplot(613)
    ax3 = fig.add_subplot(614)
    ax4 = fig.add_subplot(615)
    ax5 = fig.add_subplot(616)

    ax.scatter(data['time'], data['speed'], facecolor='none', edgecolor=color)
    ax1.scatter(data['time'], data['camber'], facecolor='none', edgecolor=color)
    ax2.scatter(data['time'], data['pressure'], facecolor='none', edgecolor=color)
    ax3.scatter(data['time'], data['fz'], facecolor='none', edgecolor=color)
    ax4.scatter(data['time'], data['slip_angle'], facecolor='none', edgecolor=color)
    ax5.scatter(data['time'], data['tsto'], facecolor='none', edgecolor=color)
    ax5.scatter(data['time'], data['tstc'], facecolor='none', edgecolor=color)
    ax5.scatter(data['time'], data['tsti'], facecolor='none', edgecolor=color)

    ax5.set_ylim(100, 200)


def raw_slip_force_plot(slip_list, force_list, color, label, x_label, y_label):
    """Scatter plot slip vs force
     User can pass color and label info as well"""
    if color is None:
        color = 'b'
    if label is None:
        label = None
    if x_label is None:
        x_label = None

    if y_label is None:
        y_label = None
    plt.scatter(slip_list, force_list, facecolor='none', edgecolors=color, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return None
