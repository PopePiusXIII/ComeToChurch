from __future__ import division
# import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
# import numpy as np
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

        self.tire_name = Ttk.StringVar(parent)
        self.data_type = Ttk.StringVar(parent)
        self.load = Ttk.StringVar(parent)
        self.camber = Ttk.StringVar(parent)
        self.pressure = Ttk.StringVar(parent)
        self.speed = Ttk.StringVar(parent)
        self.slip_sign = Ttk.StringVar(parent)
        self.tire_dict = None
        # corner.set('Choose Corner')
        # x_axis.set('x_axis')
        # y_axis.set('y_axis')
        Ttk.Button(self.entry_frame, text='Update', command=lambda: self.grid_drop_downs()).grid(row=2, column=0)
        Ttk.Button(self.entry_frame, text='Graph', command=lambda:
                   Fy.raw_slip_force_plot(self.tire_dict["Slip Angle (Steady)"][self.speed.get()][self.pressure.get()][self.camber.get()]
                                          [self.slip_sign.get()][self.load.get()],
                                          self.tire_dict["Fy (Steady)"][self.speed.get()][self.pressure.get()][self.camber.get()]
                                          [self.slip_sign.get()][self.load.get()], None, None, None, None
                                          )).grid(row=2, column=1)

    def grid_drop_downs(self):
        self.tire_dict = Fy.fy_data_collector(HarleyTires.file_paths[0])
        data_name_options = self.tire_dict.keys()
        tire_name_options = [HarleyTires.file_paths[0]]
        speed_options = self.tire_dict['Speed (Test)']
        slip_sign_options = self.tire_dict['Slip Sign (Test)']
        pressure_options = self.tire_dict['Pressure (Test)']
        camber_options = self.tire_dict['Camber (Test)']
        load_options = self.tire_dict['Load (Test)']

        Ttk.Label(self.entry_frame, text='Name').grid(row=0, column=0)
        tire_name_menu = Ttk.OptionMenu(self.entry_frame, self.tire_name, *tire_name_options)
        tire_name_menu.config(width=13)
        tire_name_menu.grid(row=1, column=0)
        Ttk.Label(self.entry_frame, text='Speed').grid(row=0, column=1)
        speed_menu = Ttk.OptionMenu(self.entry_frame, self.speed, *speed_options)
        speed_menu.config(width=4)
        speed_menu.grid(row=1, column=1)
        Ttk.Label(self.entry_frame, text='Pressure').grid(row=0, column=2)
        pressure_menu = Ttk.OptionMenu(self.entry_frame, self.pressure, *pressure_options)
        pressure_menu.config(width=12)
        pressure_menu.grid(row=1, column=2)
        Ttk.Label(self.entry_frame, text='Camber').grid(row=0, column=3)
        camber_menu = Ttk.OptionMenu(self.entry_frame, self.camber, *camber_options)
        camber_menu.config(width=6)
        camber_menu.grid(row=1, column=3)
        Ttk.Label(self.entry_frame, text='Slip Sign').grid(row=0, column=4)
        slip_sign_menu = Ttk.OptionMenu(self.entry_frame, self.slip_sign, *slip_sign_options)
        slip_sign_menu.config(width=9)
        slip_sign_menu.grid(row=1, column=4)
        Ttk.Label(self.entry_frame, text='Load').grid(row=0, column=5)
        load_menu = Ttk.OptionMenu(self.entry_frame, self.load, *load_options)
        load_menu.config(width=4)
        load_menu.grid(row=1, column=5)
        return


class PageOne(Ttk.Frame):
    def __init__(self, parent, controller):

        # FRAMES
        Ttk.Frame.__init__(self, parent)

app = HarleyTires()
app.mainloop()
