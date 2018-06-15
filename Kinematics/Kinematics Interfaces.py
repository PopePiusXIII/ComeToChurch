from __future__ import division
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import Tkinter as Ttk
import tkFileDialog
import Kinematics as Kin
import numpy as np
from collections import OrderedDict
import copy
# import Motec_Import as Motec
matplotlib.use('TkAgg')


class HarleyTires(Ttk.Tk):
    # 2d dictionary structured as [corner][point name] refer to full_car.txt for corner names and keys
    full_car = {}

    def __init__(self, *args, **kwargs):
        Ttk.Tk.__init__(self, *args, **kwargs)

        # -----------------CONTAINER FOR ALL FRAMES--------------------------
        container = Ttk.Frame(self)
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # -------------MENU BAR----------------------
        main_menu = Ttk.Menu(container)
        filemenu = Ttk.Menu(main_menu, tearoff=0)
        sim_menu = Ttk.Menu(main_menu, tearoff=0)
        filemenu.add_command(label='Open..', command=lambda: self.open())
        filemenu.add_command(label='Save as..', command=lambda: self.save_as(HarleyTires.full_car))
        filemenu.add_command(label='Export Solidworks..', command=lambda: self.export_solidworks(HarleyTires.full_car))
        sim_menu.add_command(label=Simulation.__name__, command=lambda: self.show_frame(Simulation))
        sim_menu.add_command(label='Motec Csv Import', command=lambda: self.motec_import())
        Ttk.Tk.config(self, menu=main_menu)
        main_menu.add_cascade(label='File', menu=filemenu)
        main_menu.add_cascade(label='Simulation', menu=sim_menu)

        self.frames = {}
        # -----------FRAMES/PAGES-------------------------
        for F in (Kinematics, Simulation):
            frame = F(parent=container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Kinematics)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    @staticmethod
    def save_as(dictionary):
        file_path = tkFileDialog.askopenfilename()
        file1 = open(file_path, 'w')
        file1.write('UNIVERSITY OF CENTRAL FLORIDA KINEMATIC SOFTWARE')
        file1.write('\n{}CREATED BY: HARLEY HICKS'.format(' '*10))
        file1.write('\n{}2017-2018 SEASON\n'.format(' '*13))
        file1.write('\n{}Suspension Points\n\n'.format(' '*13))
        for corner_key in dictionary.keys():
            file1.write("\n{}{}{}\n".format('-'*10, corner_key, '-'*10,))
            for key in dictionary[corner_key].keys():
                if len(key) < 15:
                    file1.write("{0} \t\t{1}\n".format(key, dictionary[corner_key][key]))
                elif len(key) >= 15:
                        file1.write("{0} \t{1}\n".format(key, dictionary[corner_key][key]))

    @staticmethod
    def export_solidworks(dictionary):
        """Solidworks takes tab delimited txt files for point clouds. So this file creates a tab delimited txt
        file of the points to make the suspension point cloud quickly"""
        file_path = tkFileDialog.askopenfilename()
        file1 = open(file_path, 'w')
        for corner_key in dictionary.keys():
            for key in dictionary[corner_key].keys():
                for num in dictionary[corner_key][key]:
                    num = num * .0254
                    file1.write("{0} ".format(num))
                file1.write("\n")
            return

    @staticmethod
    def motec_import():
        """Solidworks takes tab delimited txt files for point clouds. So this file creates a tab delimited txt
        file of the points to make the suspension point cloud quickly"""
        file_path = tkFileDialog.askopenfilename()
        Simulation.motec_data = Motec.motec_csv_import(file_path)
        return

    def open(self):
        HarleyTires.full_car = OrderedDict([])  # creating the initial full car dictionary

        file_path = tkFileDialog.askopenfilename()      # file path browser call

        with open(file_path, 'r') as suspension:   # read in file skip first 8 lines because not useful for the code
            data = suspension.readlines()[7:]

        for line in data:
            line = line.translate(None, "\n\t,")
            if line == '':
                pass
            elif line[0] == '-':  # checking for a blank line above indicating a new corner
                line = line.translate(None, "-")
                HarleyTires.full_car[line] = OrderedDict([])

            elif len(line) > 0:
                split_line = line.split(' ')    # split line up into list of strings by the spaces separating them
                name = ''  # point name from txt file
                point = []  # list that will be appended by values split by SPACES
                bracket_flag = False    # check for the start of comment separated data
                for string in split_line:
                    if '[' in string:   # if open bracket append data by split spaces until closing bracket
                        bracket_flag = True
                        string = string[1:]  # delete '[' so no float errors boii

                    if bracket_flag:
                        string = string.translate(None, '] ')
                        if any(char.isdigit() for char in string):      # check for number so can be converted to float
                            point.append(float(string))     # point list with all data appended by float
                        else:
                            point.append(string)   # no digits so point list appended by string

                    else:
                        if len(name) > 0:
                            name = name + ' ' + string
                        elif len(name) == 0:
                            name = string
                HarleyTires.full_car[HarleyTires.full_car.keys()[-1]][name] = point

        # call entry_box_update in Kinematics page to overwrite points in entry boxes
        # print HarleyTires.full_car.keys()
        Kinematics.entry_box_create_update(self.frames[Kinematics], HarleyTires.full_car, True, 'Front')
        Simulation.create_entry_wheel_disp(self.frames[Simulation], HarleyTires.full_car, True)

        return HarleyTires.full_car


class Kinematics(Ttk.Frame):
    pbind = OrderedDict([])  # pbind = point binding (ttk var) in entry box
    entry_box = copy.deepcopy(pbind)  # entry box binding/locations same structure as points and full car
    last_key = []

    def __init__(self, parent, controller):
        print 'Kinematics controller', controller
        # FRAMES
        Ttk.Frame.__init__(self, parent)        # start page container within the separate page container

        # frame to house all the buttons for changing between susp and sims
        self.button_frame = Ttk.Frame(self, height=200, width=200)
        self.button_frame.grid(row=0, column=0)

        self.graph_frame = Ttk.Frame(self, height=600, width=600)      # 3d graph frame
        self.graph_frame.grid(row=1, column=1, sticky='n')

        # entry frame where user will change the suspension points round
        self.entry_frame = Ttk.Frame(self, height=900, width=200)
        self.entry_frame.grid(row=1, column=0, sticky='n')

        # entry frame where user will change the suspension points round
        self.analysis_frame = Ttk.Frame(self, height=900, width=200)
        self.analysis_frame.grid(row=1, column=2, sticky='n')

        # front suspension plot raise
        Ttk.Button(self.button_frame, text='Front Suspension', command=lambda:
                   self.entry_box_create_update(HarleyTires.full_car, False, 'Front')
                   ).grid(row=0, column=0)
        # rear suspension plot raise
        Ttk.Button(self.button_frame, text='Rear Suspension', command=lambda:
                   self.entry_box_create_update(HarleyTires.full_car, False, 'Rear')
                   ).grid(row=0, column=1)

    def entry_box_create_update(self, dictionary, open_flag, axle):
        """creates labels and entry boxes for all widgets in performance and entry frame"""
        # -------------------------ENTRY FRAME-------------------------------
        i = 0
        if axle == 'Front':
            key1 = "Left Front"
            key2 = 'Right Front'
        elif axle == 'Rear':
            key1 = "Left Rear"
            key2 = "Right Rear"
        else:
            print 'WHAT IS THIS A SEMI?'
            return

        if key1 in self.last_key:
            pass
        else:
            self.last_key = [key1, key2]

        for corner_key in self.last_key:     # iterate through original suspension dict (should be txt file l8r)

            # loop through suspension point dictionary to collect keys and make a spot in memory for the binding
            if len(self.pbind) < len(dictionary):
                for key in dictionary.keys():
                    self.pbind[key] = OrderedDict([])
                    self.entry_box[key] = OrderedDict([])

            for key in dictionary[corner_key]:
                i += 1
                # creates entry boxes that will be used for life of the gui
                # build bindings if not established for every point in dictionary
                # covers initial opening
                if len(self.pbind[corner_key].keys()) < len(dictionary[corner_key].keys()):
                    # print'should only happen first time'
                    # print len(self.pbind[corner_key].keys()), len(dictionary[corner_key].keys())
                    # labels what the points are in entry frame
                    Ttk.Label(self.entry_frame, text=key).grid(row=i, column=0, sticky='w')
                    # filling func scope list with bindings to the individual tk.entries
                    self.pbind[corner_key][key] = Ttk.StringVar()
                    # entries and bindings est.
                    self.entry_box[corner_key][key] = (Ttk.Entry(self.entry_frame, width=23,
                                                                 textvariable=self.pbind[corner_key][key]))
                    self.entry_box[corner_key][key].insert(0, "%.3f, %.3f, %.3f" % (dictionary[corner_key][key][0],
                                                                                    dictionary[corner_key][key][1],
                                                                                    dictionary[corner_key][key][2]))

                elif open_flag:   # else retrieve from entry boxes or txt file and refill them with the new user data
                    # used when user opens from txt file
                    self.entry_box[corner_key][key].delete(0, 30)  # clear boxes
                    # refill boxes with data
                    self.entry_box[corner_key][key].insert(
                        0, "%.3f, %.3f, %.3f" % (HarleyTires.full_car[corner_key][key][0],
                                                 HarleyTires.full_car[corner_key][key][1],
                                                 HarleyTires.full_car[corner_key][key][2]))
                else:   # used when user makes changes directly in entry box
                    # use map to separate by ","
                    HarleyTires.full_car[corner_key][key] = map(float, self.pbind[corner_key][key].get().split(','))
                    self.entry_box[corner_key][key].delete(0, 30)     # clear boxes
                    # refill boxes with data
                    self.entry_box[corner_key][key].insert(
                        0, "%.3f, %.3f, %.3f" % (HarleyTires.full_car[corner_key][key][0],
                                                 HarleyTires.full_car[corner_key][key][1],
                                                 HarleyTires.full_car[corner_key][key][2]))

                self.entry_box[corner_key][key].grid(row=i, column=1)
                self.entry_box[corner_key][key].lift()
        # call plot function
        self.suspension_plot(HarleyTires.full_car, key1, key2)
        self.create_analysis_frame(HarleyTires.full_car, axle)

    def create_analysis_frame(self, dictionary, axle):
        if axle == 'Front':
            key = ["Left Front", 'Right Front', 'Performance Figures']
        elif axle == 'Rear':
            key = ["Left Rear", 'Right Rear', 'Performance Figures']
        else:
            print 'WHAT IS THIS A SEMI?'
            return
        # --------------------------------ANALYSIS FRAME-----------------------------------

        # Caster analysis Label
        Ttk.Label(self.analysis_frame, text='Caster').grid(row=0, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % Kin.two_d_vertical_angle(HarleyTires.full_car[key[0]]['Lower Out'][0:3:2],
                                                         HarleyTires.full_car[key[0]]['Upper Out'][0:3:2])
                  ).grid(row=0, column=1, sticky='w')

        # Heave Damper Length Analysis Label
        Ttk.Label(self.analysis_frame, text='Heave Damper Length').grid(row=1, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % Kin.magnitude(HarleyTires.full_car[key[0]]['Damper Rocker'],
                                              HarleyTires.full_car[key[1]]['Damper Rocker'])
                  ).grid(row=1, column=1, sticky='w')

        # Roll Damper Length Analysis Label
        Ttk.Label(self.analysis_frame, text='Roll Damper Length').grid(row=2, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % Kin.magnitude(HarleyTires.full_car[key[0]]['Roll Damper a'],
                                              HarleyTires.full_car[key[1]]['Roll Damper a'])
                  ).grid(row=2, column=1, sticky='w')

        # Kingpin Inclination Label
        Ttk.Label(self.analysis_frame, text='Kingpin Inclination').grid(row=3, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % Kin.two_d_vertical_angle(HarleyTires.full_car[key[0]]['Lower Out'][1:],
                                                         HarleyTires.full_car[key[0]]['Upper Out'][1:])
                  ).grid(row=3, column=1, sticky='w')    # Finds 2D vertical angle from steering axis to vertical

        # Kingpin offset Label
        line_len = Kin.shortest_line_to_point(HarleyTires.full_car[key[0]]['Upper Out'],
                                              HarleyTires.full_car[key[0]]['Lower Out'],
                                              HarleyTires.full_car[key[0]]['Wheel Center'])[1]
        offset = np.subtract(line_len, HarleyTires.full_car[key[0]]['Wheel Center'])
        kpo = offset[1]     # Finds shortest length from wheel center to steering axis, then takes specific components
        Ttk.Label(self.analysis_frame, text='Kingpin Offset').grid(row=4, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % kpo).grid(row=4, column=1, sticky='w')

        # Caster offset Label
        caso = offset[0]
        Ttk.Label(self.analysis_frame, text='Caster Offset').grid(row=5, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f ' % caso).grid(row=5, column=1, sticky='w')

        # Track Width Label
        distance = np.subtract(HarleyTires.full_car[key[0]]['Wheel Center'],
                               HarleyTires.full_car[key[1]]['Wheel Center'])
        ydist = distance[1]
        Ttk.Label(self.analysis_frame, text='Track Width').grid(row=6, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % ydist).grid(row=6, column=1, sticky='w')

        # Mechanical Trail Label
        intersection = Kin.three_d_vector_plane_intersection(HarleyTires.full_car[key[0]]['Lower Out'],
                                                             HarleyTires.full_car[key[0]]['Upper Out'],
                                                             [0, 1, 0], [1, 0, 0], [0, 0, 0])
        Ttk.Label(self.analysis_frame, text='Mechanical Trail').grid(row=7, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % np.subtract(intersection,
                                                                 HarleyTires.full_car[key[0]]['Wheel Center'])[0]
                  ).grid(row=7, column=1, sticky='w')

        # Scrub Radius Label
        intersection = Kin.three_d_vector_plane_intersection(HarleyTires.full_car[key[0]]['Lower Out'],
                                                             HarleyTires.full_car[key[0]]['Upper Out'], [0, 1, 0],
                                                             [1, 0, 0], [0, 0, 0])
        Ttk.Label(self.analysis_frame, text='Scrub Radius').grid(row=8, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % np.subtract(intersection,
                                                                 HarleyTires.full_car[key[0]]['Wheel Center'])[1]
                  ).grid(row=8, column=1, sticky='w')

        # Establishing Instant Center
        ic_direction, ic_point = Kin.plane_intersection_line(
            Kin.plane_equation(HarleyTires.full_car[key[0]]['Upper Fore'],
                               HarleyTires.full_car[key[0]]['Upper Aft'],
                               HarleyTires.full_car[key[0]]['Upper Out']),
            Kin.plane_equation(HarleyTires.full_car[key[0]]['Lower Fore'],
                               HarleyTires.full_car[key[0]]['Lower Aft'],
                               HarleyTires.full_car[key[0]]['Lower Out']),
            HarleyTires.full_car[key[0]]['Upper Fore'],
            HarleyTires.full_car[key[0]]['Lower Fore'])

        axis = Kin.plot_line(ic_direction, ic_point, np.linspace(0, 2, 2))

        # Instant Center Longitudinal Label
        ic_xz = Kin.three_d_vector_plane_intersection((axis[0][0], axis[1][0], axis[2][0]),
                                                      (axis[0][1], axis[1][1], axis[2][1]),
                                                      HarleyTires.full_car[key[0]]['Wheel Center'],
                                                      np.add(np.array(HarleyTires.full_car[key[0]]
                                                                      ['Wheel Center']), np.array([1, 0, 0])),
                                                      np.add(np.array(HarleyTires.full_car[key[0]]
                                                                      ['Wheel Center']), np.array([0, 0, 1])))
        # print 'xz Instant Centner %s' % ic_xz
        Ttk.Label(self.analysis_frame, text='Instant center long').grid(row=9, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f         ' % ic_xz[0]).grid(row=9, column=1, sticky='w')

        # Instant Center Lateral Label
        ic_yz = Kin.three_d_vector_plane_intersection((axis[0][0], axis[1][0], axis[2][0]),
                                                      (axis[0][1], axis[1][1], axis[2][1]),
                                                      HarleyTires.full_car[key[0]]['Wheel Center'],
                                                      np.add(np.array(HarleyTires.full_car[key[0]]
                                                             ['Wheel Center']),
                                                      np.array([0, 1, 0])),
                                                      np.add(np.array(HarleyTires.full_car[key[0]]
                                                             ['Wheel Center']),
                                                      np.array([0, 0, 1])))
        # print 'yz Instant Centner %s' % ic_yz
        Ttk.Label(self.analysis_frame, text='Instant center lat').grid(row=10, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f         ' % ic_yz[1]).grid(row=10, column=1, sticky='w')

        # Jacking height Label (This is the equivalent to the FAP)
        y_val = HarleyTires.full_car[key[2]]['Center of Gravity'][1]
        cg_plane_points = [[1, y_val, 1], [-1, y_val, 4], [-3, y_val, 6]]
        wheel_center_ground = [(HarleyTires.full_car[key[0]]['Wheel Center'][0]),
                               (HarleyTires.full_car[key[0]]['Wheel Center'][1]), 0]
        np.array(wheel_center_ground)
        jacking_height = Kin.three_d_vector_plane_intersection(wheel_center_ground,
                                                               ic_yz, cg_plane_points[0], cg_plane_points[1],
                                                               cg_plane_points[2])
        Ttk.Label(self.analysis_frame, text='Jacking Height').grid(row=11, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % jacking_height[2]).grid(row=11, column=1, sticky='w')

        # Jacking Distance Label
        Ttk.Label(self.analysis_frame, text='Jacking Distance').grid(row=12, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f' % Kin.magnitude(jacking_height, ic_yz)
                  ).grid(row=12, column=1, sticky='w')

        # Jacking Coefficient Label
        wc_jh = np.subtract(jacking_height, wheel_center_ground)
        jacking_coeff = wc_jh[2]/wc_jh[1]
        Ttk.Label(self.analysis_frame, text='Jacking Coefficient').grid(row=13, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % jacking_coeff).grid(row=13, column=1, sticky='w')

        # Anti-squat Height Label
        as_height = Kin.three_d_vector_plane_intersection(wheel_center_ground, ic_xz, [55.5, 0, 1],
                                                          [55.5, -1, 0], [55.5, -3, 11])
        Ttk.Label(self.analysis_frame, text='Anti-squat Height').grid(row=14, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f ' % as_height[2]).grid(row=14, column=1, sticky='w')

        # Pitch Coefficient Label
        wc_icxz = np.subtract(ic_xz, HarleyTires.full_car[key[0]]['Wheel Center'])
        wc_cg = np.subtract(HarleyTires.full_car[key[2]]['Center of Gravity'],
                            HarleyTires.full_car[key[0]]['Wheel Center'])
        pitch_coeff = (wc_icxz[2] / wc_icxz[0]) / (wc_cg[2] / wc_cg[0])
        # print wc_icxz, wc_cg
        Ttk.Label(self.analysis_frame, text='Pitch Coefficient').grid(row=15, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f' % pitch_coeff).grid(row=15, column=1, sticky='w')

        # Anti-squat/dive Label
        Ttk.Label(self.analysis_frame, text='Anti-squat/dive').grid(row=16, column=0, sticky='w')
        Ttk.Label(self.analysis_frame,
                  text='%.3f ' % (np.subtract(HarleyTires.full_car[key[2]]['Center of Gravity'], as_height))[2]
                  ).grid(row=16, column=1, sticky='w')

        # Small Step Motion Ratio Label
        sim_results = Kin.bump_sim(np.linspace(0, .01, 2), np.linspace(0, .00001, 2),
                                   HarleyTires.full_car, 'Bump', key[0], key[1])
        sim_eval_results = Kin.sim_evaluation(HarleyTires.full_car, sim_results, "Bump", key[0], key[1])

        Ttk.Label(self.analysis_frame, text='Bump MR').grid(row=17, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f ' % sim_eval_results[key[0]]['Bump Heave Damper MR'][0]
                  ).grid(row=17, column=1, sticky='w')

        # Small step bump roll motion ratio
        Ttk.Label(self.analysis_frame, text='Roll MR').grid(row=18, column=0, sticky='w')
        Ttk.Label(self.analysis_frame, text='%.3f ' % sim_eval_results[key[0]]['Bump Roll Damper MR'][0]
                  ).grid(row=18, column=1, sticky='w')

        return

    def suspension_plot(self, full_car_dict, *keys):
        # 3D SUSPENSION GRAPH
        key1 = keys[0]
        key2 = keys[1]
        f = Figure(figsize=(5, 5), dpi=100)
        canvas = FigureCanvasTkAgg(f, master=self.graph_frame)
        ax = Axes3D(f)
        # call suspension_plot func from Kinematics
        Kin.suspension_plot(ax, full_car_dict, True, True, key1, key2)
        canvas._tkcanvas.grid(row=0, column=0)
        return


class Simulation(Ttk.Frame):
    sim_results_ = ()     # variable to save the bump sim results
    sim_eval_results = OrderedDict()    # variable to save the bump sim evaluation results
    motec_data = OrderedDict()

    def __init__(self, parent, controller):

        # FRAMES
        Ttk.Frame.__init__(self, parent)

        self.button_frame = Ttk.Frame(self)  # frame to house all the buttons for changing between susp and sims
        self.button_frame.grid(row=0, column=0, sticky='w')

        self.entry_frame = Ttk.Frame(self)  # entry frame where user will change the suspension points round
        self.entry_frame.grid(row=1, column=0, columnspan=4)

        self.wheel_disp_button_frame = Ttk.Frame(self)  # frame for wheel displacement sim buttons
        self.wheel_disp_button_frame.grid(row=2, column=0, columnspan=4)

        self.wheel_disp_entry_frame = Ttk.Frame(self)  # frame for wheel displacement entry boxes
        self.wheel_disp_entry_frame.grid(row=3, column=0, columnspan=4)

        self.wheel_disp_batch_frame = Ttk.Frame(self)  # frame for wheel displacement entry boxes
        self.wheel_disp_batch_frame.grid(row=4, column=0, columnspan=4)

        # --------------CREATING HEAVE ENTRY BOX AND BINDINGS FOR SIM----------------
        self.motion_heave_binding = []  # range of values to use for sim wheel displacement
        Ttk.Label(self.entry_frame, text='Heave (Start, Finish, Resolution)').grid(columnspan=3)
        initial_heave = [1, 0.0001, 10]
        for i in range(0, 3, 1):
            self.motion_heave_binding.append(Ttk.DoubleVar())
            entry = Ttk.Entry(self.entry_frame, textvariable=self.motion_heave_binding[i])
            entry.delete(0, 30)
            entry.insert(0, initial_heave[i])
            entry.grid(row=1, column=i)

        # ---------------HEAVE BUTTONS--------------------
        # Button to Run bumpsim and evaluation from Kinematics.py
        Ttk.Button(self.entry_frame, text='RUN HEAVE',
                   command=lambda: self.sim_results_set(self.motion_heave_binding, 'Heave', 'Left Front',
                                                        'Right Front')).grid(row=3, column=1)
        Ttk.Button(self.button_frame, text='Kinematics', command=lambda: controller.show_frame(Kinematics)).grid()
        Ttk.Button(self.button_frame, text='Graph', command=self.graph_gui).grid(row=0, column=1)

        # ---------------- SINGLE WHEEL BUMP-----------------------------
        self.motion_bump_binding = []
        Ttk.Label(self.entry_frame, text='Bump (Start, Finish, Resolution)').grid(row=4, columnspan=3)
        initial_bump = [1, 0.0001, 10]
        for i in range(0, 3, 1):
            self.motion_bump_binding.append(Ttk.DoubleVar())
            entry = Ttk.Entry(self.entry_frame, textvariable=self.motion_bump_binding[i])
            entry.delete(0, 30)
            entry.insert(0, initial_bump[i])
            entry.grid(row=5, column=i)

        # ---------------BUMP BUTTONS--------------------
        # Button to Run bumpsim and evaluation from Kinematics.py
        Ttk.Button(self.entry_frame, text='RUN BUMP', command=lambda: self.sim_results_set(self.motion_bump_binding,
                                                                                           'Bump',
                                                                                           'Left Front', 'Right Front'
                                                                                           )).grid(row=6, column=1)
        # ----------------Wheel Displacement-----------------
        Ttk.Label(self.wheel_disp_button_frame,
                  text='Wheel Displacement ([Weight], [Spring], '
                       '[Motion Ratio], [Spring Corner], [Shims])').grid(row=0, columnspan=4)

        Ttk.Button(self.wheel_disp_button_frame, text='Run Wheel Displacement',
                   command=lambda: Kin.four_corner_wheel_displacement(HarleyTires.full_car)).grid(row=1, column=0)

        Ttk.Button(self.wheel_disp_button_frame, text='Update Boxes',
                   command=lambda: self.create_entry_wheel_disp(HarleyTires.full_car, False)).grid(row=2, column=0)

        # ----------------Wheel Displacement Batch--------------------
        Ttk.Button(self.wheel_disp_batch_frame, text='Batch Run Wheel Disp',
                   command=lambda: Kin.wheel_disp_compare_plot(HarleyTires.full_car, np.linspace(0, 1, 9),
                                                               np.linspace(0, 0.001, 9), np.array([150, 150, 150, 150]))
                   ).grid(row=3, column=0)

        Ttk.Button(self.wheel_disp_batch_frame, text='Motec Wheel Displacement',
                   command=lambda: Kin.wheel_disp_compare_plot(HarleyTires.full_car, self.motec_data['Long Accel'],
                                                               self.motec_data['Lat Accel'],
                                                               np.array([140, 140, 160, 160]), self.motec_data['Time'],
                                                               self.motec_data['Damper Pos FL'],
                                                               self.motec_data['Damper Pos FR'],
                                                               self.motec_data['Damper Pos LR'],
                                                               self.motec_data['Damper Pos RR'],
                                                               self.motec_data['Velocity']
                                                               )
                   ).grid(row=3, column=1)

    def create_entry_wheel_disp(self, dictionary, open_flag):
        i = 0   # used for placing entry boxes in columns
        # create entry boxes on initial opening of the gui
        if open_flag:
            self.pbind = OrderedDict([])
            self.entry_boxes = OrderedDict([])
            for keys in dictionary['Performance Figures'].keys():
                self.pbind[keys] = Ttk.StringVar()  # create binding for tk variables
                Ttk.Label(self.wheel_disp_entry_frame, text=keys).grid(row=3+i, column=0)
                self.entry_boxes[keys] = Ttk.Entry(self.wheel_disp_entry_frame, textvariable=self.pbind[keys], width=75)
                self.entry_boxes[keys].grid(row=3+i, column=1)
                self.entry_boxes[keys].delete(0, 75)    # clear entry box
                entry_list = dictionary['Performance Figures'][keys]
                string = ', '.join(str(x) for x in entry_list)
                self.entry_boxes[keys].insert(0, string)
                i += 1
        else:  # used when user makes changes directly in entry box
            # use map to separate by ","
            def check_ifdigit(string):
                if any(char.isdigit() for char in string):
                    return float(string)
                else:
                    return string.translate(None, ' ')
            for keys in dictionary['Performance Figures'].keys():
                dictionary['Performance Figures'][keys] = map(check_ifdigit, self.pbind[keys].get().split(','))
                self.entry_boxes[keys].delete(0, 75)  # clear entry box
                entry_list = dictionary['Performance Figures'][keys]
                string = ', '.join(str(x) for x in entry_list)
                self.entry_boxes[keys].insert(0, string)
                self.entry_boxes[keys].grid(row=3+i, column=1)
                i += 1

    def sim_results_set(self, motion_list, motion, *corners):
        """Calls simulation solvers and defines results as self.sim_eval_results
        Motion: list of displacements"""
        motion_range = []   # list of retrieved values for bump motion measured at wheel entered into entry boxes
        for i in motion_list:
            motion_range.append(i.get())
        self.sim_results = Kin.bump_sim(np.linspace(motion_range[0], motion_range[1],
                                                    int(motion_range[2])),
                                        np.linspace(0, .00001, int(motion_range[2])),
                                        HarleyTires.full_car, motion, *corners)
        self.sim_eval_results = Kin.sim_evaluation(HarleyTires.full_car, self.sim_results, motion, *corners)
        return

    # method to graphs pop up windows dood
    def graph_gui(self):
        """POP UP Window to graph selections"""
        graph_selection_app = Ttk.Tk()

        # -------------------------LABELS--------------------------
        Ttk.Label(graph_selection_app, text='Corner').grid(row=0, column=0)
        Ttk.Label(graph_selection_app, text='x axis').grid(row=1, column=0)
        Ttk.Label(graph_selection_app, text='y axis').grid(column=0, row=2)

        # ----------------------VARIABLES----------------------------
        corner = Ttk.StringVar(graph_selection_app)
        x_axis = Ttk.StringVar(graph_selection_app)
        y_axis = Ttk.StringVar(graph_selection_app)
        corner.set('Choose Corner')
        x_axis.set('x_axis')
        y_axis.set('y_axis')

        # ------------------------------OPTION MENUS-----------------------------
        # corner menu
        Ttk.OptionMenu(graph_selection_app, corner, *self.sim_eval_results.keys()
                       ).grid(row=0, column=1)
        # x menu
        Ttk.OptionMenu(graph_selection_app, x_axis, *self.sim_eval_results['Left Front'].keys()
                       ).grid(row=1, column=1)
        # y menu
        Ttk.OptionMenu(graph_selection_app, y_axis, *self.sim_eval_results['Left Front'].keys()
                       ).grid(row=2, column=1)
        # ----------------------------BUTTONS----------------------------------
        Ttk.Button(graph_selection_app, text='Graph',
                   command=lambda: Kin.scatter_plot(self.sim_eval_results[corner.get()][x_axis.get()],
                                                    self.sim_eval_results[corner.get()][y_axis.get()],
                                                    x_axis.get(), y_axis.get())).grid()

        graph_selection_app.mainloop()


app = HarleyTires()
app.geometry("1000x700")
app.mainloop()
