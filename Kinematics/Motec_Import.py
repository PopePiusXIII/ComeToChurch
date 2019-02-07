import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import Track_Gen as TrackGen


def motec_csv_quotation_remove(file_path, new_filepath):
    with open(file_path, 'r') as infile, open(new_filepath, 'w') as outfile:
        data = infile.read()
        data = data.replace('"', "")
        outfile.write(data)


def motec_csv_import(file_path):
    if file_path is None:
        return
    # opening the dat file to be sorted
    first_row = 18
    print file_path
    file_location = file_path
    # skip first 4 rows to jump strings    unpack=(True transposes the matrix)    dtype=float_ to turn it into float
    raw_data = np.loadtxt(file_location, delimiter=',', skiprows=first_row, unpack=True, dtype='float_')
    data_dict = OrderedDict([
        ('Lat Accel', raw_data[3]),
        ('Long Accel', raw_data[10]),
        ('Time', raw_data[0]),
        ('Damper Pos FL', raw_data[6]),
        ('Damper Pos FR', raw_data[17]),
        ('Damper Pos RR', raw_data[18]),
        ('Damper Pos LR', raw_data[19]),
        ('Velocity', raw_data[-5]),
    ])

    return data_dict
motec_csv_quotation_remove("C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPad.csv",
                           "C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPadNoCommas.csv")
x=motec_csv_import("C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPadNoCommas.csv")

print x
plt.subplot(1, 1, 1)
track = TrackGen.track_plotter(x['Velocity'], x['Lat Accel'], .1)
plt.axis('equal')
plt.scatter(track[0], track[1], facecolor='None', edgecolors='r')
plt.show()

plt.subplot(4, 1, 2)
plt.plot(x['Velocity'])

plt.subplot(4, 1, 3)
plt.plot(x['Lat Accel'])

plt.subplot(4, 1, 4)
plt.plot(track[3])
