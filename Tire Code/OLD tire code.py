from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def tire_code(file_paths, number_to_be_compared):

    # figuring out how many files to run through using file name length as the guide

    # setting up temporary storage inside the function
    # tire/slip angle sign/psi/camber/load
    filtered_time = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_fz = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_sa = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_sr = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_fy = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_fx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    friction_fy = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    friction_fx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    friction_mx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    friction_mz = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_fy = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_fx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_mx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_mz = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_sa = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    norm_sr = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_mz = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_mx = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    filtered_temperature = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
    pac_coeff = [[[[[[] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]

    print np.shape(filtered_time), 'filter time len'
    # starting to read the raw data excel file, will cycle based on number to be compared. Max 2 min 1

    for t_compare in range(0, number_to_be_compared, 1):

        # opening the excel file to be sorted
        first_row = 10
        file_location = file_paths[t_compare]
        run = np.loadtxt(file_location, skiprows=first_row)
        last_row = len(run)
        print "Tire number %s, Length of Data = %s \n File Path %s" % (t_compare + 1, last_row, file_paths[t_compare])

        # storing data from the excel sheet
        # skip first five rows to jump words and titles in sheet
        # clears everything at the start of every for loop but the list storing data at the end
        speed_column = 1
        sa_column = 3
        sr_column = 20
        camber_column = 4
        fy_column = 9
        fx_column = 8
        mz_column = 12
        mx_column = 11
        tsti_column = 16
        tstc_column = 17
        tsto_column = 18
        time_column = 0
        psi_column = 7
        fz_column = 10
        fz_graph = []
        psi_graph = []
        sa_graph = []
        sr_graph = []
        camber = []
        camber_data_length = [0] * 5
        temp_avg = []
        tsti = []
        tstc = []
        tsto = []
        speed = []
        fz_raw = [[0] for i in range(5)]
        psi_raw = [[0] for i in range(5)]
        sa_raw = [[0] for i in range(5)]
        sr_raw = [[0] for i in range(5)]
        mx_raw = [[0] for i in range(5)]
        fy_raw = [[0] for i in range(5)]
        fx_raw = [[0] for i in range(5)]
        mz_raw = [[0] for i in range(5)]
        slip_rate = [[0] for i in range(5)]
        temp_raw = [[0] for i in range(5)]
        raw_time = [[0] for i in range(5)]
        time_test = []
        camber_test = []
        filter_time = []
        filter_psi = []
        psi_test = []
        fz_test = []
        sa_test = []
        sr_test = []
        fy_test = []
        mz_test = []
        mx_test = []
        fx_test = []
        slip_rate_test = []
        filter_slip_rate = []
        filter_fz = []

        # store main parameters of data to use in filtering
        for i in range(0, last_row - first_row, 1):
            speed.append(run[i][speed_column])
            camber.append(run[i][camber_column])
            tstc.append(run[i][tstc_column])
            tsto.append(run[i][tsto_column])
            tsti.append(run[i][tsti_column])
            temp_avg.append((tsti[i] + tstc[i] + tsto[i])/3)
            time_test.append(run[i][time_column])
            camber_test.append(run[i][camber_column])
            psi_test.append(run[i][psi_column])
            fz_test.append(run[i][fz_column])
            sa_test.append(run[i][sa_column])
            sr_test.append(run[i][sr_column])
            mz_test.append(run[i][mz_column])
            mx_test.append(run[i][mx_column])
            fx_test.append(-run[i][fx_column])
            fy_test.append(-run[i][fy_column])
            if i > 3:
                slip_rate_test.append((sa_test[-1] - sa_test[-4]) / (time_test[-1] - time_test[-4]))
            else:
                slip_rate_test.append(0)

        print 'stop 1', len(speed)
        # filtering out transition loads, cold tires, speed outside the constant, transition camber
        for cambercount in range(0, 5, 1):
            for y in range(0, len(speed), 1):

                if (120 < temp_avg[y] < 160) and ((cambercount - .5) < abs(camber[y]) < (cambercount + .5)) and (24 < speed[y] < 26):
                    # storing the raw data from the excel sheet into vectors
                    fz_raw[cambercount].append(fz_test[y])
                    sa_raw[cambercount].append(sa_test[y])
                    sr_raw[cambercount].append(sr_test[y])
                    fy_raw[cambercount].append(fy_test[y])
                    fx_raw[cambercount].append(fx_test[y])
                    mx_raw[cambercount].append(mx_test[y])
                    psi_raw[cambercount].append(psi_test[y])
                    mz_raw[cambercount].append(mz_test[y])
                    raw_time[cambercount].append(time_test[y])
                    temp_raw[cambercount].append(temp_avg[y])

                    # setting up the slip rate for later filtering
                    if len(sa_raw[cambercount]) > 3:
                        slip_rate[cambercount].append((sa_raw[cambercount][-1] - sa_raw[cambercount][-4]) / (raw_time[cambercount][-1] - raw_time[cambercount][-4]))
                    else:
                        slip_rate[cambercount].append(0)
                    camber_data_length[cambercount] += 1

        print 'camber data length', camber_data_length
        stop = 0
        for i in range(0, 5, 1):
            stop += camber_data_length[i]
        print 'stop 2', stop
        # starting to assign values for filtered_fz loads within a certain range
        r = 0
        loads = [-50, -100, -150, -200, -250, -350]
        pressures = [8, 10, 12, 14]
        for x in range(0, 5, 1):
            i = camber_data_length[x]
            for y in range(0, i, 1):
                if fz_raw[x][y] != 0:   # and (3.8 < slip_rate[x][y] < 4.2)
                    # sorting based on load
                    index = 0
                    for i in loads:
                        # print "fz check", fz_raw[x][y], index
                        if int(.9 * i) >= fz_raw[x][y] >= int(1.1 * i):
                            fz_graph.append(index)
                            break
                        index += 1
                    if index == len(loads):
                        continue

                    index = 0
                    for i in pressures:
                        # print'psi check', psi_raw[x][y], index
                        if .95 * i < psi_raw[x][y] < 1.05 * i:
                            psi_graph.append(index)
                            break
                        index += 1
                    if index == len(pressures):
                        del fz_graph[-1]
                        continue

                    # sorting based on positive or negative slip angle
                    if 0 <= sa_raw[x][y]:
                        sa_graph.append(1)
                    else:
                        sa_graph.append(0)

                    if 0 < sr_raw[x][y]:
                        sr_graph.append(1)
                    else:
                        sr_graph.append(0)

                    # Tire/psi/camber/load
                    filtered_sa[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(sa_raw[x][y])
                    filtered_fy[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(fy_raw[x][y])
                    filtered_fx[t_compare][sr_graph[r]][psi_graph[r]][x][fz_graph[r]].append(fx_raw[x][y])
                    filtered_mz[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(mz_raw[x][y])
                    filtered_fz[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(fz_raw[x][y])
                    filtered_mx[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(mx_raw[x][y])
                    friction_fy[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(((fy_raw[x][y]) / -(fz_raw[x][y])))
                    friction_fx[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(((fx_raw[x][y]) / -(fz_raw[x][y])))
                    friction_mz[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(((mz_raw[x][y]) / -(fz_raw[x][y])))
                    friction_mx[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(((mx_raw[x][y]) / -(fz_raw[x][y])))
                    filtered_time[t_compare][sa_graph[r]][psi_graph[r]][x][fz_graph[r]].append(raw_time[x][y])
                    filtered_temperature[t_compare][sr_graph[r]][psi_graph[r]][x][fz_graph[r]].append(temp_raw[x][y])
                    filter_time.append(raw_time[x][y])
                    filter_psi.append([psi_raw[x][y]])
                    filter_fz.append(fz_raw[x][y])
                    filter_slip_rate.append(slip_rate[x][y])
                    filtered_sr[t_compare][sr_graph[r]][psi_graph[r]][x][fz_graph[r]].append(int(10*sr_raw[x][y]))
                    filtered_fx[t_compare][sr_graph[r]][psi_graph[r]][x][fz_graph[r]].append(fx_raw[x][y])

                    r += 1

        # CALCULATING FRICTIONS AND NORMS
        for i in range(t_compare + 1):
            for j in range(max(sa_graph)+1):
                for k in range(max(psi_graph)+1):
                    for l in range(5):
                        for m in range(max(fz_graph)+1):
                            if len(friction_fy[i][j][k][l][m]) > 0:
                                temp_max_fy = (max(friction_fy[i][j][k][l][m]))
                                temp_max_mz = (max(friction_mz[i][j][k][l][m]))
                                temp_max_mx = (max(friction_mx[i][j][k][l][m]))
                                temp_min_fy = (min(friction_fy[i][j][k][l][m]))
                                temp_min_mz = (min(friction_mz[i][j][k][l][m]))
                                temp_min_mx = (min(friction_mx[i][j][k][l][m]))
                                temp_max_fx = (max(friction_fx[i][j][k][l][m]))
                                temp_min_fx = (min(friction_fx[i][j][k][l][m]))
                                for n in range(0, len(friction_fy[i][j][k][l][m]), 1):

                                    if temp_max_fy != 0:

                                        if abs(temp_min_fy) > temp_max_fy:
                                            norm_fy[i][j][k][l][m].append(friction_fy[i][j][k][l][m][n] / -temp_min_fy)
                                        else:
                                            norm_fy[i][j][k][l][m].append(friction_fy[i][j][k][l][m][n] / temp_max_fy)

                                        if abs(temp_min_fy) > temp_max_fy:
                                            norm_mz[i][j][k][l][m].append(friction_mz[i][j][k][l][m][n] / -temp_min_mz)
                                        else:
                                            norm_mz[i][j][k][l][m].append(friction_mz[i][j][k][l][m][n] / temp_max_mz)

                                        if abs(temp_min_mx) > temp_max_mx:
                                            norm_mx[i][j][k][l][m].append(friction_mx[i][j][k][l][m][n] / -temp_min_mx)
                                        else:
                                            norm_mx[i][j][k][l][m].append(friction_mx[i][j][k][l][m][n] / temp_max_mx)

                                        if abs(temp_min_fx) > temp_max_fx:
                                            norm_fx[i][j][k][l][m].append(friction_fx[i][j][k][l][m][n] / -temp_min_fx)
                                        else:
                                            norm_fx[i][j][k][l][m].append(friction_fx[i][j][k][l][m][n] / temp_max_fx)

        # FITTING CORNER STIFFNESS
        small_sa = []
        small_fy = []
        small_fx = []
        small_sr = []
        cornering_stiffness_fy = [[[[[[0] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]
        cornering_stiffness_fx = [[[[[[0] for i in range(6)] for i in range(5)] for i in range(4)] for i in range(2)] for i in range(number_to_be_compared)]

        for i in range(t_compare + 1):
            for j in range(max(sa_graph)+1):
                for k in range(max(psi_graph)+1):
                    for l in range(5):
                        for m in range(max(fz_graph)+1):
                            for n in range(len(filtered_sa[i][j][k][l][m])):
                                if -2 < filtered_sa[i][j][k][l][m][n] < 2:
                                    small_sa.append(filtered_sa[i][j][k][l][m][n])
                                    small_fy.append(filtered_fy[i][j][k][l][m][n])
                            print i, j, k, l, m, n, 'fy'
                            if len(small_sa) > 0:
                                print'length 1', len(small_sa)
                                cornering_stiffness_fy[i][j][k][l][m] = (np.polyfit(small_sa, small_fy, 1))
                                small_sa = []
                                small_fy = []

        # Cornering stiffness fx
        for i in range(t_compare + 1):
            for j in range(max(sa_graph) + 1):
                for k in range(max(psi_graph) + 1):
                    for l in range(5):
                        for m in range(max(fz_graph) + 1):
                            for n in range(len(filtered_sr[i][j][k][l][m])):
                                if -.05 < filtered_sr[i][j][k][l][m][n] < .05:
                                    small_sr.append(filtered_sr[i][j][k][l][m][n])
                                    small_fx.append(filtered_fx[i][j][k][l][m][n])
                            print i, j, k, l, m, n, 'fx'
                            if len(small_sr) > 0:
                                print'length', len(small_sr)
                                cornering_stiffness_fx[i][j][k][l][m] = (np.polyfit(small_sr, small_fx, 1))
                                small_sr = []
                                small_fx = []

        # NORMING SLIP Angle
        for i in range(t_compare + 1):
            pass
        for j in range(max(sa_graph)+1):
            for k in range(max(psi_graph)+1):
                for l in range(5):
                    for m in range(max(fz_graph)+1):
                        for n in range(len(filtered_sa[i][j][k][l][m])):
                            if j == 1:
                                mue_fy = max(filtered_fy[i][j][k][l][m]) / -min(filtered_fz[i][j][k][l][m])
                                norm_sa[i][j][k][l][m].append((filtered_sa[i][j][k][l][m][n] / 57.3 * abs(cornering_stiffness_fy[i][j][k][l][m][0])) / (mue_fy * -filtered_fz[i][j][k][l][m][n]))
                            else:
                                mue_fy = -min(filtered_fy[i][j][k][l][m]) / -min(filtered_fz[i][j][k][l][m])
                                norm_sa[i][j][k][l][m].append((filtered_sa[i][j][k][l][m][n] / 57.3 * abs(cornering_stiffness_fy[i][j][k][l][m][0])) / (mue_fy * -filtered_fz[i][j][k][l][m][n]))
                        print i, j, k, l, m
        # NORMING SLIP RATIO
        for i in range(t_compare + 1):
            for j in range(max(sa_graph) + 1):
                for k in range(max(psi_graph) + 1):
                    for l in range(5):
                        for m in range(max(fz_graph) + 1):
                            for n in range(len(filtered_sr[i][j][k][l][m])):
                                if j == 1:
                                    mue_fx = max(filtered_fx[i][j][k][l][m]) / -min(filtered_fz[i][j][k][l][m])
                                    norm_sr[i][j][k][l][m].append((filtered_sr[i][j][k][l][m][n] / 57.3 * abs(
                                        cornering_stiffness_fx[i][j][k][l][m][0])) / (
                                                                      mue_fx * -filtered_fz[i][j][k][l][m][n]))
                                else:
                                    mue_fx = -min(filtered_fx[i][j][k][l][m]) / -min(filtered_fz[i][j][k][l][m])
                                    norm_sr[i][j][k][l][m].append((filtered_sr[i][j][k][l][m][n] / 57.3 * abs(
                                        cornering_stiffness_fx[i][j][k][l][m][0])) / (
                                                                      mue_fx * -filtered_fz[i][j][k][l][m][n]))
                            print i, j, k, l, m

        print 'finished'

    return filtered_sa, filtered_sr, filtered_fy, filtered_fx, filtered_fz, filtered_mx, filtered_mz, friction_fy, friction_mz, friction_mx, norm_fy, norm_mz, norm_mx, filtered_time, filtered_temperature,time_test, camber_test, psi_test, fz_test, tsti, tstc, tsto, speed, filter_slip_rate, slip_rate_test, sr_test, sa_test, pac_coeff, norm_sa, norm_sr, norm_fx

'''
# example how to use in another module
# tire number(0-n)/slip sign(-0, +1)/psi(0-3)/camber(0-4)/load(0-6)
data = tire_code('C:\Users\Harley\Documents\FSAE\Suspension\KR17\TTC Code\Round 3\Goodyear 7.0-20.5(7.0 wheel, ALL psi).xlsx', '', 1)
slip = data[0]
forces = data[8]
temp = data[12]

tire_coefficients = Tire_Fitting.least_square(slip[2][1][2][0][5], forces[2][1][2][0][5])
r2 = R_Squared.r_squared(forces[2][1][2][0][5], slip[2][1][2][0][5], tire_coefficients)

numbers = np.linspace(-12, 12, 50)
y = Tire_Fitting.model(numbers, tire_coefficients)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plt.plot(numbers, Tire_Fitting.model(numbers, tire_coefficients))

ax1.scatter(slip[2][1][2][0][5], forces[2][1][2][0][5], c=temp[2][1][2][0][5], cmap='viridis' 'ro', label='%s' % r2)
plt.legend()
plt.show()
'''