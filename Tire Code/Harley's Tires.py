from __future__ import division
# import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
# import numpy as np
from collections import OrderedDict
import Tkinter as Ttk
# import Tire_Fitting
from tkFileDialog import askopenfilename
import Fy as Fy


class HarleyTires(Ttk.Tk):
    # must have none in file_paths so drop down menus will not break on page one before a tire path is chosen
    file_paths = [None]

    def __init__(self, *args, **kwargs):
        Ttk.Tk.__init__(self, *args, **kwargs)
        container = Ttk.Frame(self)

        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # -------------MENU BAR----------------------
        main_menu = Ttk.Menu(container)
        filemenu = Ttk.Menu(main_menu, tearoff=0)
        sim_menu = Ttk.Menu(main_menu, tearoff=0)
        filemenu.add_command(label='Add Tire', command=lambda: self.open())
        Ttk.Tk.config(self, menu=main_menu)
        main_menu.add_cascade(label='File', menu=filemenu)
        main_menu.add_cascade(label='Simulation', menu=filemenu)

        self.frames = {}

        for F in (StartPage, PageOne):
            frame = F(container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def open(self):
        """Provides the user with a gui to choose a file path from their directory
        """
        # overwrites None if the file path is the first to be chosen
        if self.file_paths[0] is None:
            self.file_paths[0] = askopenfilename()

        # appends to the list of filepaths chosen by the user
        else:
            self.file_paths.append(askopenfilename())

        return


class StartPage(Ttk.Frame):
    def __init__(self, parent, controller):

        # ------------FRAMES-----------------
        Ttk.Frame.__init__(self, parent)

        # displays tire's personal file path along with its name
        self.name_frame = Ttk.Frame()
        self.name_frame.grid(row=0, column=0)

        # entry frame where the user will choose the camber, load, etc to be evaluated
        self.entry_frame = Ttk.Frame()
        self.entry_frame.grid(row=1, column=0)

        self.tire_name = [Ttk.StringVar()]
        self.data_type = [Ttk.StringVar()]
        self.load = [Ttk.StringVar()]
        self.camber = [Ttk.StringVar()]
        self.pressure = [Ttk.StringVar()]
        self.speed = [Ttk.StringVar()]
        self.slip_sign = [Ttk.StringVar()]
        self.tire_dict = OrderedDict()
        self.graph_bool = [Ttk.IntVar()]
        # corner.set('Choose Corner')
        # x_axis.set('x_axis')
        # y_axis.set('y_axis')
        self.update = Ttk.Button(self.entry_frame, text='Update', command=lambda: self.grid_drop_downs())
        self.update.grid(row=2, column=0)
        self.graph = Ttk.Button(self.entry_frame, text='Graph', command=lambda: self.graph_func())
        self.graph.grid(row=2, column=1)

    def graph_func(self):
        colors = ['r', 'b', 'g']
        x_data=[]
        y_data=[]
        for i in range(0, len(self.tire_name)):
            if self.graph_bool[i].get() == 1:
                x_data.append(self.tire_dict[self.tire_name[i].get()]["Slip Angle (Steady)"]
                               [self.speed[i].get()][self.pressure[i].get()][self.camber[i].get()]
                               [self.slip_sign[i].get()][self.load[i].get()])
                y_data.append(self.tire_dict[self.tire_name[i].get()]["Fy (Steady)"][self.speed[i].get()]
                               [self.pressure[i].get()][self.camber[i].get()]
                               [self.slip_sign[i].get()][self.load[i].get()])
        Fy.raw_slip_force_plot(x_data, y_data, colors, None, None, None)
        return

    def grid_drop_downs(self):
        self.update.destroy()
        self.graph.destroy()

        self.tire_name.append(Ttk.StringVar())
        self.data_type.append(Ttk.StringVar())
        self.load.append(Ttk.StringVar())
        self.camber.append(Ttk.StringVar())
        self.pressure.append(Ttk.StringVar())
        self.speed.append(Ttk.StringVar())
        self.slip_sign.append(Ttk.StringVar())
        self.graph_bool.append(Ttk.IntVar())
        tire_number = len(HarleyTires.file_paths)
        self.tire_dict[HarleyTires.file_paths[tire_number-1]] = Fy.fy_data_collector(HarleyTires.file_paths[tire_number-1])
        tire_name_options = HarleyTires.file_paths
        speed_options = self.tire_dict[HarleyTires.file_paths[tire_number-1]]['Speed (Test)']
        slip_sign_options = self.tire_dict[HarleyTires.file_paths[tire_number-1]]['Slip Sign (Test)']
        pressure_options = self.tire_dict[HarleyTires.file_paths[tire_number-1]]['Pressure (Test)']
        camber_options = self.tire_dict[HarleyTires.file_paths[tire_number-1]]['Camber (Test)']
        load_options = self.tire_dict[HarleyTires.file_paths[tire_number-1]]['Load (Test)']
        Ttk.Label(self.entry_frame, text='Name').grid(row=tire_number, column=0)
        tire_name_menu = Ttk.OptionMenu(self.entry_frame, self.tire_name[tire_number-1], *tire_name_options)
        tire_name_menu.config(width=13)
        tire_name_menu.grid(row=tire_number, column=0)
        Ttk.Label(self.entry_frame, text='Speed').grid(row=tire_number, column=1)
        speed_menu = Ttk.OptionMenu(self.entry_frame, self.speed[tire_number-1], *speed_options)
        speed_menu.config(width=4)
        speed_menu.grid(row=tire_number, column=1)
        Ttk.Label(self.entry_frame, text='Pressure').grid(row=tire_number, column=2)
        pressure_menu = Ttk.OptionMenu(self.entry_frame, self.pressure[tire_number-1], *pressure_options)
        pressure_menu.config(width=12)
        pressure_menu.grid(row=tire_number, column=2)
        Ttk.Label(self.entry_frame, text='Camber').grid(row=tire_number, column=3)
        camber_menu = Ttk.OptionMenu(self.entry_frame, self.camber[tire_number-1], *camber_options)
        camber_menu.config(width=6)
        camber_menu.grid(row=tire_number, column=3)
        Ttk.Label(self.entry_frame, text='Slip Sign').grid(row=tire_number, column=4)
        slip_sign_menu = Ttk.OptionMenu(self.entry_frame, self.slip_sign[tire_number-1], *slip_sign_options)
        slip_sign_menu.config(width=9)
        slip_sign_menu.grid(row=tire_number, column=4)
        Ttk.Label(self.entry_frame, text='Load').grid(row=tire_number, column=5)
        load_menu = Ttk.OptionMenu(self.entry_frame, self.load[tire_number-1], *load_options)
        load_menu.config(width=4)
        load_menu.grid(row=tire_number, column=5)
        Ttk.Label(self.entry_frame, text='Graph').grid(row=tire_number, column=6)
        graph_bool = Ttk.Checkbutton(self.entry_frame, variable=self.graph_bool[tire_number-1])
        graph_bool.grid(row=tire_number, column=6)
        self.update = Ttk.Button(self.entry_frame, text='Update', command=lambda: self.grid_drop_downs())
        self.update.grid(row=tire_number+1, column=0)
        self.graph = Ttk.Button(self.entry_frame, text='Graph', command=lambda: self.graph_func())
        self.graph.grid(row=tire_number+1, column=1)

        return


class PageOne(Ttk.Frame):
    def __init__(self, parent, controller):

        # FRAMES
        Ttk.Frame.__init__(self, parent)

app = HarleyTires()
app.mainloop()
