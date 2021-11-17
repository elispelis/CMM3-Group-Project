#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 19 14:36:05 2021
@author: Dean, Elis
"""

#Import relevant modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

print("Loading...") #Print loading screen to let user know the code is still loading
start_time = time.time() #Start time to count how long program runs

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
N = 2 ** 17  # Number of particles
D = 0.1  # diffusivity
Nx = Ny = 64 #Euler grid size
circle_x = 0 #circles centre x coordinate
circle_y = 0 #circles centre y coordinate

# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

x = np.random.uniform(x_min, x_max, size=N) #x-positions
y = np.random.uniform(y_min, y_max, size=N) #y-positions

phi1 = np.ones(N)  # Array of ones for np.where function for plot type
phi0 = np.zeros(N)  # Array of zeros for np.where function for plot type
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_colormap', ['r', 'lime', 'b'], 64)  # colormap for graphing

PlotType = 0 #Plot either diffusive patch or non-zero velocity (choose 1 or 0)

if PlotType == 0:
    phi = np.where(np.sqrt((x+circle_x) ** 2 + (y-circle_y) ** 2) < 0.3, phi1, phi0) #create circle for diffusive patch

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
    avrphi = np.rot90(np.reshape(avrphi, [Nx, Ny]))
    return avrphi

#given a coordinate, tells us what nearest velocity vector is
def get_velocities(x, y): 
    x_coordinates = np.floor((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length-1)).astype(int) #same indexing
    y_coordinates = np.floor((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length-1)).astype(int) #as avrphi function
    #empty arrays to receive velocity data
    x_velocities = np.empty(shape=N) 
    y_velocities = np.empty(shape=N)
    for i in range(N): #turns our two vel arrays into a 1D array
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = vel[velocity_index][0]
        y_velocities[i] = vel[velocity_index][1]
    return x_velocities, y_velocities 

# def plot_data(): #Function to plot our data
#     plt.subplot(2,5,int(i/55)+1) #Gives 10 subplots
#     print(str(int(i/55)*10)+ "%") #Prints loading percentage
#     avphi = getavrphimesh(x, y) #Assigns our avphi values to plot
#     plt.title('t= '+str(i/1000)+'s') #Shows the time where the subplot was plotted
#     plt.imshow(avphi, cmap=cmap, extent=(x_min, x_max, y_min, y_max)) #Plot using imshow to display data as an image

fig = plt.figure()
axes = []

for i in np.linspace(0, int(t_max/dt), int(t_max/dt)+1):
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection
    #Bounce particles off boundary walls: if coord is beyond boundary, make it be as much as it exceeded boundary
    x = np.where(x > x_max, 2 * x_max - x, x)
    x = np.where(x < x_min, 2 * x_min - x, x) 
    y = np.where(y > y_max, 2 * y_max - y, y) 
    y = np.where(y < y_min, 2 * y_min - y, y)
    #Plot the data
    if i%55 == 0:
        print(str(int(i/55)*10)+ "%") #Prints loading percentage
        avphi = getavrphimesh(x, y) #Assigns our avphi values to plot
        axes.append(fig.add_subplot(2,5,int(i/55)+1))
        im = plt.imshow(avphi, cmap=cmap, extent=(x_min, x_max, y_min, y_max)) #Plot using imshow to display data as an image
        axes[-1].set_title('t= '+str(i/1000)+'s') #Shows the time where the subplot was plotted

fig.tight_layout()
plt.subplots_adjust(right=0.8)
plt.clim(0, 1) #Colourbar limits
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Concentration (ϕ)') #Gives colourbar a title
plt.suptitle('Task A: Advection & Diffusion') #Title of plot
print("--- %s seconds ---" % (time.time() - start_time)) #Shows code running time
plt.show() #Show plot