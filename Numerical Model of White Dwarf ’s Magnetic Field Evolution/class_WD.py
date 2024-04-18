import numpy as np
from tqdm import tqdm
import methods as met
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# constant values in cgs unit:
M_sun  = 1.98847e33      # in gram
r_sun  = 6.957e10        # in cm
c      = 2.9979e10       # in cm / seconds
yrs    = 3.1556e7        # in seconds


class White_Dwarf:
    def __init__(self, mass):
        # All data are stored in cgs unit:
        self.age = 0
        self.mass = mass * M_sun
        self.r_surface_size = 0


    # load data file to the white dwarf:
    def load_data(self, directory, file_type =  "general"):
        if file_type == "general":
            time, R_core, _, _, R_out  =  np.genfromtxt(directory, unpack = True, delimiter= ",", skip_header = 1)
            self.time =  time * yrs
            self.r_convection_size = R_out * r_sun
            self.r_core_size = R_core * r_sun

        if file_type == "diffusivity":
            r_WD, _ = np.genfromtxt(directory , unpack = True, delimiter= "")
            self.r_surface_size = r_WD * r_sun


    # build raidial grid for R(r) computation:
    def initiate_radial_grid(self, npt):
        if self.r_convection_size.all() == 0:
            raise ValueError ("Load the diffusivity file to get WD radius first")
        
        else:
            self.npt = npt
            self.r_grid = np.linspace(min(self.r_surface_size), max(self.r_surface_size), self.npt)
            self.dr =  self.r_grid[1] - self.r_grid[0]

    

    # read diffusivity and evlove with time:
    def evolve_field(self, folder_directory, r_out_frac = 0.2, inter_steps = 100, convection_boost = False, boost_factor = 1):

        # inter step defines number of step in between each time point
        # default is 100

        # build initial condiction first:
        # default 0.2 R_WD convectionsize
        initial_R = met.build_initial(self.r_grid, self.r_core_size[0], r_out_frac * max(self.r_surface_size))

        # Build empty slots to store evolution of field
        self.R_evo = np.zeros(shape=[self.files_read + 1, self.npt])
        self.R_evo[0] = initial_R
        self.R_past = np.zeros(shape=[self.files_read, self.npt])



        range_list = range(1, self.files_read + 1)
        for i in tqdm(range_list):
            rr, sigma = np.genfromtxt(folder_directory + f"/file{i}.dat" , unpack = True, delimiter= "")

            rr = rr[::-1] * r_sun
            eta = c**2 / (4*np.pi * sigma[::-1])
            # interpolation:
            eta_interp = np.interp(self.r_grid, rr, eta)


            if convection_boost == True:
                eta_interp = met.conv_boost(eta_interp, self.r_grid, mag = boost_factor,\
                                             solidcore_size= self.r_core_size[i -1],\
                                             conv_zone_size= self.r_convection_size[i - 1])


            step = inter_steps
            if i == self.files_read:
                dt=dt
            else:
                dt = (self.time[i] - self.time[i - 1]) / step

            self.R_past[i-1] = met.evo_R(self.R_evo[i - 1], eta_interp, dt, step, self.dr, self.r_grid)[-1]
            self.R_evo[i] = self.R_past[i -1]
        
        print(f"Evolution completed. Total number of steps:{inter_steps * self.files_read}")

        return self.R_evo
    



    # change existing field into magnetic trace lines:
    def get_psi(self):
        
        self.psi = np.zeros([self.files_read, self.npt, self.npt])

        xg = np.zeros([self.files_read, self.npt, self.npt])
        yg = np.zeros([self.files_read, self.npt, self.npt])

        for j in tqdm(range(self.files_read)):
            self.psi[j], xg[j], yg[j] = met.convert_R_to_psi(self.r_grid, self.R_evo[j])

        print ("Conversion completed.")

        # returns all the quantities at every time, 
        # set xy-coord for object
        self.sandbox_xgrid = xg[0]
        self.sandbox_ygrid = yg[0]

        return  self.psi



    def read_files(self, folder_directory):
        # first to check how many files in data package:
        self.files_read = 0
        while True:
            try:
                file_name = folder_directory + f"/file{self.files_read + 1}.dat"
                _, _ = np.genfromtxt(file_name , unpack = True, delimiter= "")
                self.files_read += 1

            except FileNotFoundError:
                print("Total files read:", self.files_read)
                break


    def produce_field_line_img(self, folder_directory, grid_pt = 512, line_density = 30, line_color = "r"):
        
        # create array to store results:
        self.psi = np.zeros([self.files_read, self.npt, self.npt])
        xg = np.zeros([self.files_read, grid_pt, grid_pt])
        yg = np.zeros([self.files_read, grid_pt, grid_pt])

        for k in tqdm(range(self.files_read)):
            # get field line
            self.psi[k], xg[k], yg[k] = met.convert_R_to_psi(self.r_grid, self.R_evo[k])

            # producing the image:
            fig1, ax2 = plt.subplots(figsize = [7.7, 6])
            kk2 = ax2.contour(yg[k] / max(self.r_surface_size), xg[k] / max(self.r_surface_size),\
                              self.psi[k], line_density, colors  = line_color)
            kk1 = ax2.contourf(yg[k] / max(self.r_surface_size), xg[k] / max(self.r_surface_size), \
                                  self.psi[k], line_density, cmap=plt.cm.bone)
            
            # tracing out the core zone size
            surface_Line = Circle((0, 0), 1, color='black', fill=False, label = "surface")
            core_Line = Circle((0, 0), self.r_core_size[k]/ max(self.r_surface_size), color='black', linestyle = "--",  fill=False, label = "core")
            ax2.add_patch(surface_Line)
            ax2.add_patch(core_Line)
            ax2.legend()
            # Add the contour line levels to the colorbar
            cbar1 = fig1.colorbar(kk1)
            cbar1.add_lines(kk2)
   
            ax2.set_title('t =' + str( np.round((self.time[k] / yrs / 1e9), 2 ))+ "Gyrs")
            ax2.set_xlabel(r"$R/R_{WD}$")
            ax2.set_ylabel(r"$R/R_{WD}$")

            fig_name = folder_directory + f"/fig{k}.png"
            plt.savefig(fig_name)
            plt.close()

