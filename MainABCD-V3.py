'''Diffusion & Advection Simulator

This code simulates fluid diffusion and advection by taking in  
user-specified inputs and plots this for a visual overview.
It shows the concentration in each grid and the diffusion can be
observed over time with each subplot.

The simulation has a loading message from 0 to 100% to assure the
user that the program is still running. Once it is complete, the
relevant plots and simulation run-time will be displayed.

It was written by Dean Sammanthan, Elis Bright, Samir Mohamed,
Zhang Li and Finn Murphy.

(c) 2021 Dean Sammanthan
(c) 2021 Elis Bright
(c) 2021 Samir Mohamed
(c) 2021 Zhang Li
(c) 2021 Finn Murphy
'''

# Import relevant packages and modules 
import sys # Used to exit the execution of the code when certain user inputs are made
import time # Used to show the user the time a certain piece of code took to run
from tkinter import * # Used for our GUI
import matplotlib as mpl # Used for our colourmap in our plots
import matplotlib.pyplot as plt  # Used to plot the relevant plots
import numpy as np  # Used to manipulate arrays and perform matrix operations
from scipy.optimize import curve_fit # Used to fit the errors to a curve

root = Tk() # Initialise Tkinter
root.title("Diffusion & Advection Interface") # Give our GUI window a title

# Creating variables for Task A & D sections in GUI (2D Problems)
DD = StringVar()  # Float variable
NxNy = StringVar()  # Integer variable
NN = StringVar()  # Integer variable
phi_limm = StringVar() # Float variable
x_minn = StringVar()  # Integer variable
x_maxx = StringVar()  # Integer variable
y_minn = StringVar()  # Integer variable
y_maxx = StringVar()  # Integer variable
t_maxx = StringVar()  # Float variable
circle_x = StringVar()  # Float variable
circle_y = StringVar()  # Float variable
circle_radius = StringVar() # Float variable

# Creating variables for Task B section in GUI (1D Problem)
avg_nn = StringVar()  # Integer variable
avg_nnn1 = StringVar() # Integer variable
avg_nnn2 = StringVar() # Integer variable
avg_nnn3 = StringVar() # Integer variable
avg_nnn4 = StringVar() # Integer variable
avg_nnn5 = StringVar() # Integer variable
h_step1 = StringVar() # Float variable
h_step2 = StringVar() # Float variable
h_step3 = StringVar() # Float variable
h_step4 = StringVar() # Float variable
h_step5 = StringVar() # Float variable

# Creating window to input the variables
label1 = Label(root, text="Please input the desired parameters here if running 2D problems or Chemical Spill").grid(row=1, column=0)

label2 = Label(root, text="Diffusivity (Applies to 1D Problem):").grid(row=2, column=0)
Entry(root, textvariable=DD).grid(row=2, column=1)
Entry(root, textvariable=DD).insert(0, "0.1")

label3 = Label(root, text="Simulation time:").grid(row=3, column=0)
Entry(root, textvariable=t_maxx).grid(row=3, column=1)
Entry(root, textvariable=t_maxx).insert(0, "0.5")

label4 = Label(root, text="Timestep h (0.001 default, can be changed in dropdown):").grid(row=4, column=0)
dtt_options = StringVar(root)
dtt_options.set("0.001")
dtt = OptionMenu(root, dtt_options, "0.001", "0.002", "0.003", "0.004",
                "0.005").grid(row=4, column=1)

label5 = Label(root, text="Grid size (Nx,Ny):").grid(row=5, column=0)
Entry(root, textvariable=NxNy).grid(row=5, column=1)
Entry(root, textvariable=NxNy).insert(0, "64")

label7 = Label(root, text="Circular Patch Specifications:").grid(row=7, column=0)

label8 = Label(root, text="Center x-value:").grid(row=7, column=1)
Entry(root, textvariable=circle_x).grid(row=8, column=1)
Entry(root, textvariable=circle_x).insert(0, "0.4")

label9 = Label(root, text="Center y-value:").grid(row=7, column=2)
Entry(root, textvariable=circle_y).grid(row=8, column=2)
Entry(root, textvariable=circle_y).insert(0, "0.4")

label10 = Label(root, text="Circle Radius:").grid(row=7, column=3)
Entry(root, textvariable=circle_radius).grid(row=8, column=3)
Entry(root, textvariable=circle_radius).insert(0, "0.1")

label12 = Label(root, text="Domain limits:").grid(row=9, column=0)

label13 = Label(root, text="Lower x limit:").grid(row=9, column=1)
Entry(root, textvariable=x_minn).grid(row=10, column=1)
Entry(root, textvariable=x_minn).insert(0, "-1")

label14 = Label(root, text="Upper x limit:").grid(row=9, column=2)
Entry(root, textvariable=x_maxx).grid(row=10, column=2)
Entry(root, textvariable=x_maxx).insert(0, "1")

label15 = Label(root, text="Lower y limit:").grid(row=9, column=3)
Entry(root, textvariable=y_minn).grid(row=10, column=3)
Entry(root, textvariable=y_minn).insert(0, "-1")

label16 = Label(root, text="Upper y limit:").grid(row=9, column=4)
Entry(root, textvariable=y_maxx).grid(row=10, column=4)
Entry(root, textvariable=y_maxx).insert(0, "1")

label17 = Label(root, text=" ").grid(row=11, column=0)

label18 = Label(root, text="Please input the desired parameters here if running 1D problem:").grid(row=12, column=0)

label19 = Label(root, text="Number of simulations:").grid(row=13, column=0)
Entry(root, textvariable=avg_nn).grid(row=13, column=1)
Entry(root, textvariable=avg_nn).insert(0, "5")

label20= Label(root, text="Specify Several Number of Particles to be Considered (Your integer input multiplied by 64):").grid(row=14, column=0)
label21 = Label(root, text="Tip: Fill 'empty' fields with '0'.").grid(row=14, column=1)
Entry(root, textvariable=avg_nnn1).grid(row=14, column=2)
Entry(root, textvariable=avg_nnn1).insert(0, "20")

Entry(root, textvariable=avg_nnn2).grid(row=14, column=3)
Entry(root, textvariable=avg_nnn2).insert(0, "100")

Entry(root, textvariable=avg_nnn3).grid(row=14, column=4)
Entry(root, textvariable=avg_nnn3).insert(0, "250")

Entry(root, textvariable=avg_nnn4).grid(row=14, column=5)
Entry(root, textvariable=avg_nnn4).insert(0, "1000")

Entry(root, textvariable=avg_nnn5).grid(row=14, column=6)
Entry(root, textvariable=avg_nnn5).insert(0, "2500")

label22 = Label(root, text="Specify Several Time Steps to be Considered:").grid(row=15, column=0)
label23 = Label(root, text="Tip: Fill 'empty' fields with '0'.").grid(row=15, column=1)
Entry(root, textvariable=h_step1).grid(row=15, column=2)
Entry(root, textvariable=h_step1).insert(0, "0.01")

Entry(root, textvariable=h_step2).grid(row=15, column=3)
Entry(root, textvariable=h_step2).insert(0, "0.02")

Entry(root, textvariable=h_step3).grid(row=15, column=4)
Entry(root, textvariable=h_step3).insert(0, "0")

Entry(root, textvariable=h_step4).grid(row=15, column=5)
Entry(root, textvariable=h_step4).insert(0, "0")

Entry(root, textvariable=h_step5).grid(row=15, column=6)
Entry(root, textvariable=h_step5).insert(0, "0")

label24 = Label(root, text=" ").grid(row=16, column=0)

label25 = Label(root, text="Please input parameters here when running 2D problems:").grid(row=17, column=0)

label26 = Label(root, text="Number of Particles:").grid(row=18, column=0)
Entry(root, textvariable=NN).grid(row=18, column=1)
Entry(root, textvariable=NN).insert(0, "150000")

label30 = Label(root, text="Concentration limit in chemical spill:").grid(row=19, column=0)
Entry(root, textvariable=phi_limm).grid(row=19, column=1)
Entry(root, textvariable=phi_limm).insert(0, "0.3")

label27 = Label(root, text="Choose velocity type").grid(row=20, column=0)
vel_options = StringVar(root)
vel_options.set("Click here to choose type")
vel = OptionMenu(root, vel_options, "Zero velocity", "Read from velocity file").grid(row=20, column=1)

label28 = Label(root, text="Choose problem type").grid(row=21, column=0)
ic_options = StringVar(root)
ic_options.set("Click here to choose type")
ic = OptionMenu(root, ic_options, "For 2D Problem (Diffusive patch)",
                "For 2D Problem (Rectangles)", "For Chemical Spill Problem",
                "For 1D Problem").grid(row=21, column=1)

label29 = Label(root, text=" ").grid(row=22, column=0)

Confirm = Button(root, text="Confirm inputs", fg="black", command=root.destroy).grid(row=23, column=1)
root.mainloop()

# Takes in variables from inputs in the GUI
D = float(DD.get()) # Diffusivity
x_min = float(x_minn.get()) # Minimum bound of plot domain in x-axis
x_max = float(x_maxx.get()) # Maximum bound of plot domain in x-axis
y_min = float(y_minn.get()) # Minimum bound of plot domain in y-axis
y_max = float(y_maxx.get()) # Maximum bound of plot domain in y-axis
t_max = float(t_maxx.get()) # Simulation time
dt = float(dtt_options.get()) # Timestep
Nx = Ny = int(NxNy.get()) # Grid size
N = int(NN.get()) # Number of particles
phi_lim = float(phi_limm.get()) #Concentration threshold for task D
circ_x = float(circle_x.get()) # Circles centre x coordinate
circ_y = float(circle_y.get()) # Circles centre y coordinate
circ_r = float(circle_radius.get()) # Circles radius

# Task B specific inputs
avg_n = int(avg_nn.get())
N_list = [64*int(avg_nnn1.get()),64*int(avg_nnn2.get()),64*int(avg_nnn3.get()),64*int(avg_nnn4.get()),64*int(avg_nnn5.get())] # List of number of particles for which to run the 1D simulation

for i in range(len(N_list)): # Remove zero values from the list
    try:
        N_list.remove(0) 
    except:
        continue
N_list.sort(reverse = True) # Sort from highest to lowest.

h = [float(h_step1.get()), float(h_step2.get()), float(h_step3.get()), float(h_step4.get()), float(h_step5.get())] # List of time steps for which to run the 1D simulation

for i in range(len(h)): # Remove zero values from the list
        try:
            h.remove(0)
        except:
            continue    
h.sort() # Sort from lowest to highest

# Errors section - Raises errors when inputs are contradictory or problematic
    
if D == 0:
    raise ValueError("You have inputted zero diffusivity. Particles will not diffuse unless a positive non-zero value is used.")

if ic_options.get() == "For 2D Problem (Diffusive patch)" or ic_options.get() == "For Chemical Spill Problem": # Errors raised when circular patch extends outside the limits assigned by the user

    if circ_x < x_min or circ_x > x_max:
        raise ValueError("Your circular patch's centre lies outside the limits you have assigned.")
    
    if circ_y < y_min or circ_y > y_max:
        raise ValueError("Your circular patch's centre lies outside the limits you have assigned.")
    
    if (circ_x - circ_r) < x_min or (circ_x + circ_r) > x_max:
        raise ValueError("Your circular patch lies outside the limits you have assigned. Try using a smaller radius or moving the centre.")
    
    if (circ_y - circ_r) < y_min or (circ_y + circ_r) > y_max:
        raise ValueError("Your circular patch lies outside the limits you have assigned. Try using a smaller radius or moving the centre.")

if ic_options.get() == "For 1D Problem": # The number of particles list determines whether an error plot can be shown. At least one step size is needed in order to run the simulation.
    
    if avg_n == 0:
        raise ValueError("You have input 0 number of simulations, so no simulation will run. Minimum value that can be used is 1 (though the RMSE values will not be averaged in that case).")

    if len(N_list) == 0:
        raise ValueError("You have no inputs for the number of particles. Please insert at least 1 to produce a plot of concentration vs. position. Inserting at least 3 produces a meaningful error plot.")
    
    if len(N_list) == 1:
        choice = input("Error plot will not be shown. You have only 1 'number of particles' inputted. Do you want to continue? (y/n) ")
        
        if choice.strip() != "y" and choice.strip() != "n": # In case the user responds with something other than "y" or "n", prompt them again. The .strip() removes leading and trailing spaces from user input.
            print() # Empty line to help with readability
            print("You have input '", choice,"'. Please input either 'y' or 'n'. The script will run automatically if this is done again.")
            choice = input("Error plot will not be shown. You have only 1 'number of particles' inputted. Do you want to continue? (y/n) ")
            
            if choice.strip() != "y" and choice.strip() != "n": # In case the user again responds with something other than "y" or "n", run their inputs anyway.
                choice = "y"
            
        if choice.strip() == "n": # The .strip() removes leading and trailing spaces from user input
            print() # Empty line to help with readability
            print("Understood! Terminating script. Feel free to run again.")
            sys.exit() # Ends script without raising an error
        if choice.strip() == "y":
            print("Understood! Continuing as requested...")
    
    if len(N_list) == 2:
        choice = input("A trivial error plot joining 2 'number of particles' will be shown because you have only 2 'number of particles' inputted. Do you want to continue? (y/n) ")
        
        if choice.strip() != "y" and choice.strip() != "n": # In case the user responds with something other than "y" or "n", prompt them again.
            print() # Empty line to help with readability
            print("You have input '", choice,"'. Please input either 'y' or 'n'. The script will run automatically if this is done again.")
            choice = input("A trivial error plot joining 2 'number of particles' will be shown because you have only 2 'number of particles' inputted. Do you want to continue? (y/n) ")
            
            if choice.strip() != "y" and choice.strip() != "n": # In case the user again responds with something other than "y" or "n", run their inputs anyway.
                choice = "y"
                
        if choice.strip() == "n":
            print() # Empty line to help with readability
            print("Understood! Terminating script. Feel free to run again.")
            sys.exit() # Ends script without raising an error
        if choice.strip() == "y":
            print("Understood! Continuing as requested...")
    
    if len(h) == 0:
        raise ValueError("You have no inputs for the step sizes you would like to consider. Please insert at least 1 to produce plots of concentration vs. position and error vs. 'number of particles'.")

# End of errors section

# Create a mesh and find the average phi values within it
def getavrphimesh(x, y):
    x_gran = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * Nx).astype(int)
    y_gran = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * Ny).astype(int)
    grancoord = np.column_stack((x_gran, y_gran))
    unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
    avrphi = np.bincount(ids, phi) / count
    avrphi = np.delete(avrphi, [0, 1])
    avrphi = np.rot90(np.reshape(avrphi, [Nx, Ny]))
    avrphi = np.reshape(avrphi, [Nx, Ny])
    return avrphi

# Given a coordinate, tells us what nearest velocity vector is
def get_velocities(x, y):
    x_coordinates = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length - 1)).astype(int)
    y_coordinates = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length - 1)).astype(int)  
    # Empty arrays to receive velocity data
    x_velocities = np.empty(shape=N)
    y_velocities = np.empty(shape=N)
    for i in range(N):  # Turns our two vel arrays into a 1D array
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = vel[velocity_index][0]
        y_velocities[i] = vel[velocity_index][1]
    return x_velocities, y_velocities

# If condition to run Task A
if ic_options.get() == "For 2D Problem (Diffusive patch)" or ic_options.get() == "For 2D Problem (Rectangles)":
    print("Loading 2D Problem...")  # Print loading screen to let user know the code is still loading
    start_time = time.time()  # Start time to count how long program runs

    # Assign variable of file name to be read
    database_file = 'velocityCMM3.dat'
    # Read data from file and process the data from it by rearranging the array
    pos = np.genfromtxt(database_file, usecols=(0, 1))  # Position array
    vel = np.genfromtxt(database_file, usecols=(2, 3))  # Velocity array
    row_length = np.sqrt(len(pos)).astype(int)
    column_length = 4  # Only 4 columns in velocity file

    x = np.random.uniform(x_min, x_max, size=N)  # X-positions
    y = np.random.uniform(y_min, y_max, size=N)  # Y-positions

    phi1 = np.ones(N)  # Array of ones for np.where function for plot type
    phi0 = np.zeros(N)  # Array of zeros for np.where function for plot type
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_colormap', ['r', 'lime', 'b'],
                                                        64)  # Colormap for graphing

    if ic_options.get() == "For 2D Problem (Diffusive patch)":
        phi = np.where(np.sqrt((x - circ_x) ** 2 + (y - circ_y) ** 2) < circ_r, phi1,
                       phi0) # Create circle for diffusive patch
    else:
        phi = np.where(x < 0, phi1, phi0) # Create separation between above and below x-axis

    fig = plt.figure()
    axes = []
    div = int(t_max/dt/9) # Used to set subplot interval

    for i in np.linspace(0, int(t_max / dt)-1, int(t_max / dt)):
        if vel_options.get() == "Read from velocity file":
            v_x, v_y = get_velocities(x, y)
        else:
            v_x = v_y = 0
        x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) # Lagrange Diffusion and advection
        y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) # Lagrange Diffusion and advection
        # Bounce particles off boundary walls: if coord is beyond boundary, make it be as much as it exceeded boundary
        x = np.where(x > x_max, 2 * x_max - x, x)
        x = np.where(x < x_min, 2 * x_min - x, x)
        y = np.where(y > y_max, 2 * y_max - y, y)
        y = np.where(y < y_min, 2 * y_min - y, y)
        # Plot the data
        if i % div == 0:
            print(str(int(i / div) * 10) + "%") # Prints loading percentage
            avphi = getavrphimesh(x, y) # Assigns our avphi values to plot
            axes.append(fig.add_subplot(2, 5, int(i / div) + 1))
            im = plt.imshow(avphi, cmap=cmap, 
                            extent=(x_min, x_max, y_min, y_max)) # Plot using imshow to display data as an image
            axes[-1].set_title('t= ' + str(round(i*dt,3)) + 's') # Shows the time where the subplot was plotted

    fig.tight_layout() # Layout type of the plot
    plt.subplots_adjust(right=0.8) # Adjust spacing of subplots
    plt.clim(0, 1) # Colourbar limits
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # Colourbar axes
    cbar = fig.colorbar(im, cax=cbar_ax) # Show colourbar
    cbar.set_label('Concentration (ϕ)') # Gives colourbar a title
    plt.suptitle('Task A: Advection & Diffusion') # Title of plot
    print("100% Simulation complete") # Print completion message
    print("---Total time taken: %s seconds ---" % (time.time() - start_time)) # Shows code running time
    plt.show() # Show plot

#Elif condition to run task D
elif ic_options.get() == "For Chemical Spill Problem":
    print("Loading Chemical Spill...")  # Print loading screen to let user know the code is still loading
    start_time = time.time()  # Start time to count how long program runs
    
    # Assign variable of file name to be read
    database_file = 'velocityCMM3.dat'
    # Read data from file and process the data from it by rearranging the array
    pos = np.genfromtxt(database_file, usecols=(0, 1))  # Position array
    vel = np.genfromtxt(database_file, usecols=(2, 3))  # Velocity array
    row_length = np.sqrt(len(pos)).astype(int)
    column_length = 4  # Only 4 columns in velocity file
    
    x = np.random.uniform(x_min, x_max, size=N)  # X-positions
    y = np.random.uniform(y_min, y_max, size=N)  # Y-positions
    
    marker = np.zeros((Nx,Ny)) # Coordintes with all 0
    phi1 = np.ones(N)  # Array of ones for where function
    phi0 = np.zeros(N)  # Array of zeros for where function
    cmap = mpl.colors.ListedColormap(["red", "blue"])  # Colormap for graphing
    bounds = [0,phi_lim,1] # Ticks for colorbar
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    phi = np.where(np.sqrt((x - circ_x) ** 2 + (y - circ_y) ** 2) < circ_r, phi1,
                   phi0)  # Create circle for diffusive patch
        
    fig = plt.figure() # Initialise plot
    axes = []
    div = int(t_max/dt/9) # Used to set subplot interval

    for i in np.linspace(0, int(t_max/dt)-1, int(t_max/dt)):
        if vel_options.get() == "Read from velocity file":
            v_x, v_y = get_velocities(x, y)
        else:
            v_x = v_y = 0
        x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
        y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
        # Bounce particles off boundary walls: if coord is beyond boundary, make it be as much as it exceeded boundary
        x = np.where(x > x_max, 2 * x_max - x, x)
        x = np.where(x < x_min, 2 * x_min - x, x) 
        y = np.where(y > y_max, 2 * y_max - y, y) 
        y = np.where(y < y_min, 2 * y_min - y, y)

        avphi = getavrphimesh(x, y) # Get phi data
        affected = np.where(avphi>phi_lim,1,0) # All points above phi_lim given value of 1
        marker += affected # Save affected values to marker array
    
        # Plot the data
        if i%div == 0:
            marker = np.where(marker>=1,1,0) # Turn all values equal or above 1 to 1
            print(str(int(i/div)*10)+ "%") # Prints loading percentage
            axes.append(fig.add_subplot(2,5,int(i/div)+1)) # Add subplot
            im = plt.imshow(marker, cmap=cmap, 
                            extent=(x_min, x_max, y_min, y_max)) # Plot using imshow to display data as an image
            axes[-1].set_title('t= '+ str(round(i*dt,3))+'s') # Shows the time where the subplot was plotted

    fig.tight_layout() # Layout type of the plot
    plt.subplots_adjust(right=0.8) # Adjust spacing of subplots
    plt.clim(0, 1) # Colourbar limits
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # Colourbar axes
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks = bounds) # Show colourbar
    cbar.set_label('Concentration (ϕ)') # Gives colourbar a title
    plt.suptitle('Task D: Advection & Diffusion') # Title of plot
    print("100% Simulation complete") # Print completion message
    print("---Total time taken: %s seconds ---" % (time.time() - start_time)) # Shows code running time
    plt.show() # Show plot

# Elif condition to run task B
elif ic_options.get() == "For 1D Problem":

    start_time2 = time.time() # Start time to count how long program runs
    
    rmse_v = [] # Empty array to store rmse values that will be averaged later on

    class TaskB: # Create class to run task B
    
        def find_nearest(self, array, value): # Returns the index of the closest number to a certain value in a list
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        def square(self, list): # Squares each element in a list (used to square the error list)
            return [i ** 2 for i in list]
            
        def __init__(self, N, h, avg_n, D): # The constructor of the class
            
            self.ref = np.genfromtxt('reference_solution_1D.dat') # Get the reference solution
            
            #Details
            self.t_max = 0.2 # Simulation time in seconds
            self.dt = h  # Step Size
            self.D = D  # Diffsivity
            self.Nx = 64 # Euler Grid Size
            self.N = N # Number of particles for which to run the simulation
            self.avg_n = avg_n # Number of times for which to run the simulation for each configuration of time step and number of particles
            
            # Domain size
            self.x_min = -1 # Lower bound for x-values
            self.x_max = 1 # Upper bound for x-values
        
            for i in range(self.avg_n): # Looping the each configuration of time step and number of particles "avg_n" times
                
                self.run_numb = i + 1 # Counter that is used to print the run/iteration number
                
                # Functions defined within the class
                self.initial_values() 
                self.sort_values() 
                self.loop() 
                self.compute() 
                self.loop2()
                
        # Initialise the simulation with arrays that will be used later on along with getting the reference solution 
        def initial_values(self):
            self.ones = np.ones(self.N)  # Array of ones for where function
            self.zeros = np.zeros(self.N)  # Array of zeros for where function
            self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)  # Initial x-positions
            self.phi = np.where(self.x <= 0, self.ones, self.zeros) # Give x-coordinates phi values
            
        # Sort the arrays made in the initial_values function 
        def sort_values(self): 
            self.x, self.phi = zip(*sorted(zip(self.x,self.phi))) # Sorts x into ascending order
            self.x_phi = np.column_stack((self.x,self.phi)) # Stacks x against phi
            
        # Manipulate the phi values by incorporating the relevant diffusion equation
        def loop(self): 
            for i in np.arange(0, self.t_max, self.dt):
                self.x_phi[:,0] += np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=self.N) # Apply diffusion calulation
                self.x_phi[:,0][self.x_phi[:,0] < self.x_min] = self.x_min  # Any values in x thats under x_min replace to x_min
                self.x_phi[:,0][self.x_phi[:,0] > self.x_max] = self.x_max # Any values in x thats over x_max replace to x_max
       
        # Process the outputs of the loop function
        def compute(self): 
            self.avrphi = np.array([])    
            self.x_phi = self.x_phi[self.x_phi[:,0].argsort()] # Sort new x_phi into ascending
            self.phi_splits = np.split(self.x_phi[:,1], self.Nx) # Split phi values so average can be taken
        
        # Generate the final array for concentration and compare to reference solution to determine RMSE
        def loop2(self): 
            for i in range(self.Nx): # Calculate average at Nx points
                average	= np.cumsum(self.phi_splits[i])[-1]/len(self.phi_splits[i]) # Takes the sum of phi.splits and divides by its length to get average phi at each of the 64 grid points
                self.avrphi=np.append(self.avrphi, average)
            
            # Begin error analysis
            self.x_val = np.linspace(self.x_min,self.x_max, self.Nx) # Evenly distribute Nx points from the lower x-bound to the the upper x-bound
            self.ref_x_ind = [] # Empty array to carry indices of the relevant x-positions that are closest to those found in the ones calculated in the loop functions
            self.ref_y =[] # Empty array to carry the corresponding concentrations for each of the indices
            for j in range(self.x_val.size): # Loop for Nx (essentially size of the x_val array)
                d = self.ref[:,0] # Reference concentrations
                a = self.x_val[j] # Value to determine what the closest reference value to is
                s = self.find_nearest(d, a) # Find nearest x-value from reference compared to the points at which we have phi
                self.ref_x_ind.append(s) # Add to list of x-value indices
    
            for c in range(self.Nx): # Get the corresponding concentration from the reference solution using the list of x-value indices closest to each of the 64 grid points
                z = self.ref[:,1][self.ref_x_ind[c]] # Gets the concentration from the reference solution for the relevant indices
                self.ref_y.append(z) # Add value to the empty array created earlier
            
            self.error_list = [] # Empty array to carry the absolute error at each of the Nx points
            
            for k in range(self.Nx): # For each of the Nx points, take the difference between the reference and computed concentration values and add it to the empty array
                error = abs(self.ref_y[k]-self.avrphi[k]) # Error calculation
                self.error_list.append(error) # Add error to empty array created earlier
            
            self.square_error = self.square(self.error_list) # This follows from the calculation from RMSE, which requires that the errors are squared
            self.f = sum(self.square_error) # The squared errors are added
            self.rmse = np.sqrt(self.f/(self.f.size)) # The square root of the sum of the squared errors divided by the number of errors obtained, yielding RMSE
            rmse_v.append(self.rmse) # Append this value to the empty array created before the definition of the TaskB class (The line above the definition of this class)
            
            # Let the user know the status of the execution of the code by printing the following.
            print("For run ",self.run_numb,", with ",self.N, " particles and " , self.dt, " time step, the root mean square error is ",self.rmse) 
    
    # Plot the concentrations for each configuration of time step and number of particles
    def plot(ref, x_min, x_max, Nx, f_avrphis): 
        x = np.linspace(x_min, x_max, Nx) # Define plot limits and x-grid quantity  
        fig, ax = plt.subplots(1) # Create the first plot
        ax.set_prop_cycle('color',plt.cm.rainbow(np.linspace(0,1,len(N_list)*len(h)))) # Define the colourmap that determines how the color of the plots varies as more plots are added (Never repeats colours)
        for i in range(len(f_avrphis)): # f_avrphis is defined further down in the code as an array which carries ALL the coordinates for ALL the configurations of time step and number of particles
            phi_int = f_avrphis[i] # Get the coordinates of the ith configuration
            for z in range(len(phi_int)):
                phi2_int = phi_int[z] # Get the concentration of zth entry
                plt.plot(x, phi2_int, marker='.', label='dT = %f and N = %.0f' %(h[i], N_list[z])) # Plot our values and label them with time step and number of particles
        
        plt.plot(ref[:, 0], ref[:, 1], color='Black', label = 'Reference at t = 0.2 s', linewidth=4.0, alpha=0.4) # Plot reference solution (with thicker line but a little transparent)   
        plt.title('Concentration vs Position at t = 0.2s') # Plot title
        plt.xlabel('Position') # X-axis label
        plt.ylabel('Concentration') # Y-axis label
        plt.grid() # Add a grid to improve readability
        plt.legend() # Add a legend according to the labels and colours already assigned in the plt.plot command
    
    f_avrphis = [] # Empty array to carry the concentration values for ALL configurations  
    
    for i in h:
       avrphis = [] # Empty array to hold the concentration values for configurations of same time step 
       for j in N_list:
           avrphis.append(TaskB(j, i, avg_n, D).avrphi) # Add the concentration values to the empty array just created
       f_avrphis.append(avrphis) # Adds the concentration values to the empty f_avrphis array for ALL time steps and ALL number of particles    
    
    plot(np.genfromtxt('reference_solution_1D.dat'), -1, 1, 64, f_avrphis) # Plot the reference solution along with f_avrphis
    
    rmse_avg = [] # Empty array for average RMSE values (averaged over avg_n runs)
    
    avg_holder = [] # Hold average RMSE values for each iteration (for all time steps)
    
    for z in range(0,len(h)): # Repeats for all time steps
        for i in range(0, len(N_list)): # Repeats for all number of particles
            rmse_holder = [] # Holds RMSE values temporarily for each iteration
            for j in range(0, avg_n):
                counter = j + avg_n * i + z * avg_n * len(N_list) # Skips through rmse_v to add all concentration values for the same number of particles 
                rmse_holder.append(rmse_v[counter]) # Collects the values as described in the line above
            entry_average = sum(rmse_holder)/len(rmse_holder) # Average across avg_n values
            avg_holder.append(entry_average)
    
    # Plotting Error Vs Number of particles for constant step size (*All step sizes are considered)    
    
    if len(N_list) > 1: # Only produces an error plot if there is sufficient number of inputs for number of particles

        def fit_func(N, a, b): # Function to be fitted to
            return a*N**b
        
        fig, ax = plt.subplots(1) # Open a new figure for the Error vs Number of Particles plot
        
        for j in range(len(h)):
            pl_rmse = [] # Array for rmse values that will be plotted
            for i in range(len(N_list)):
                pl_rmse.append(avg_holder[i + j * len(N_list)])
            
            params = curve_fit(fit_func, N_list, pl_rmse) # Adapted from https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting - Fits our data to the function defined above
            [a, b] = params[0]
        
            fit_RMSE = [] # Empty array to carry fitted RMSE function 
        
            for z in N_list: # Getting values for the fitted RMSE solution at the same number of particles
                fit_RMSE_int = fit_func(z, a, b)
                fit_RMSE.append(fit_RMSE_int) 
        
            ax.set_xscale('log') # Set the x-axis to a logarithmic scale
            ax.set_yscale('log') # Set the y-axis to a logarithmic scale 
            plt.xlabel('Number of Particles')
            plt.ylabel('RMSE')
            plt.plot(N_list,pl_rmse, marker='.', label='Averaged RMSE over %0.d runs for %f step size' %(avg_n, h[j])) # Plotting calculated RMSE
            plt.plot(N_list,fit_RMSE, label = '"Fitted RMSE for %f stepsize" = %.3f*N^(%.3f)' %(h[j],a, b), linestyle='dashed') # Plotting fitted RMSE
            plt.title('RMSE vs Number of Particles at t = 0.2s')
            plt.legend(loc = 'lower left') #Add a legend

    print("Simulation complete")
    print("---Total time taken: %s seconds ---" % (time.time() - start_time2)) # Stop the "timer" and print the time taken for the code to execute
    plt.show()