"""
Created on Tue Oct 19 14:36:05 2021

@author: dean
"""

#Import relevant modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#Assign variable of file name to be read
database_file = 'velocityCMM3.dat'

#Read data from file and process the data from it by rearranging the array
pos = np.genfromtxt(database_file, usecols=(0, 1)) #position array
vel = np.genfromtxt(database_file, usecols=(2, 3)) #velocity array
row_length = np.sqrt(len(pos)).astype(int) 
column_length = 4 #only 4 columns in velocity file

#Details
t_max = 0.5  # simulation time in seconds
dt = 0.001  # step size
N = 2 ** 16  # Number of particles
D = 0.01  # diffusivity
Nx = Ny = 64 #Euler grid size 

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

x = np.random.uniform(x_min, x_max, size=N) #x-positions
y = np.random.uniform(y_min, y_max, size=N) #y-positions

phi1 = np.ones(N)  # Array of ones for where function
phi0 = np.zeros(N)  # Array of zeros for where function
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_colormap', ['r', 'g', 'b'], 64)  # colormap for graphing

PlotType = 1 #Plot either diffusive patch or non-zero velocity (choose 1 or 0)

if PlotType == 0:
    phi = np.where(np.sqrt(x ** 2 + y ** 2) < 0.3, phi1, phi0) #create circle for diffusive patch

if PlotType == 1:
    phi = np.where(x < 0, phi1, phi0) #create separation between above and below x-axis


# create a mesh and find the average phi values within it
def getavrphimesh(x, y):
    x_gran = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * Nx).astype(int)
    y_gran = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * Ny).astype(int) 
    grancoord = np.column_stack((x_gran, y_gran)) 
    unq, ids, count = np.unique(grancoord, return_inverse=True, return_counts=True, axis=0)
    avrphi = np.bincount(ids, phi)/count
    avrphi = np.delete(avrphi, [0, 1])
    avrphi = np.reshape(avrphi, [Nx, Ny])
    return avrphi

def get_velocities(x, y): #given a coordinate, tells us what nearest velocity vector is
    x_coordinates = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length-1)).astype(int) #same indexing
    y_coordinates = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length-1)).astype(int) #as avrphi function
    x_velocities = np.empty(shape=N) #empty arrays to receive velocity data
    y_velocities = np.empty(shape=N)
    for i in range(N): #turns our two vel arrays into a 1D array
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = vel[velocity_index][0]
        y_velocities[i] = vel[velocity_index][1]
    return x_velocities, y_velocities 

def plot_data(): #function to plot our data
    avphi = getavrphimesh(x, y)
    plt.imshow(avphi, cmap=cmap, extent=(x_min, x_max, y_min, y_max)) #plot using imshow to display data as an image
    plt.title('insert title here') #title of plot
    plt.colorbar() #colour map legend
    plt.show() #plot

for i in np.arange(0, (t_max+dt), dt):
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    #Bounce particles off boundary walls: if coord is beyond boundary, set it 
    #to be as much as it exceeded boundary (bounce)
    x = np.where(x > x_max, 2 * x_max - x, x)
    x = np.where(x < x_min, 2 * x_min - x, x) 
    y = np.where(y > y_max, 2 * y_max - y, y) 
    y = np.where(y < y_min, 2 * y_min - y, y)
    #plot the data
    plot_data()
