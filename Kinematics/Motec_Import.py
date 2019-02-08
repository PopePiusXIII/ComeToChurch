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
        ('Lat Accel', raw_data[2]),
        ('Long Accel', raw_data[10]),
        ('Time', raw_data[0]),
        ('Damper Pos FL', raw_data[6]),
        ('Damper Pos FR', raw_data[17]),
        ('Damper Pos RR', raw_data[18]),
        ('Damper Pos LR', raw_data[19]),
        ('Velocity', raw_data[-5]),
        ('GPS_Longitude', raw_data[-10]),
        ('GPS_Latitude', raw_data[-11]),
    ])

    return data_dict
motec_csv_quotation_remove("C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPad.csv",
                           "C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPadNoCommas.csv")
x = motec_csv_import("C:\\Users\\harle\\Documents\\ComeToChurch\\Testing Data\\16EnduroUSFPadNoCommas.csv")

plt.subplot(2, 1, 1)
x1, y1, radius = TrackGen.track_gen_gps(x['GPS_Latitude'], x['GPS_Longitude'], 150)
plt.axis('equal')
plt.scatter(x1[:-20], y1[:-20], facecolor='None', c=radius, cmap='hsv')

plt.subplot(2, 1, 2)
track = TrackGen.track_gen(x['Velocity'], x['Lat Accel'], .01)
plt.axis('equal')
plt.scatter(track[0], track[1], facecolor='None', c=track[3], cmap='hsv')
plt.show()

plt.subplot(4, 1, 2)
plt.plot(x['Velocity'])

plt.subplot(4, 1, 3)
plt.plot(x['Lat Accel'])

plt.subplot(4, 1, 4)
plt.plot(track[3])
