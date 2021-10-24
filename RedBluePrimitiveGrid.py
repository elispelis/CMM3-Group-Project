# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:12:31 2021

@author: Samir Elsheikh
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:43:33 2021

@author: Samir Elsheikh
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:11:11 2021

@author: Samir Elsheikh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Assign variable of file name to be read
read_file = 'velocityCMM3.dat'

#Read data from file and process the data from it by rearranging the array
pos = np.genfromtxt(read_file, usecols=(0, 1))
vel = np.genfromtxt(read_file, usecols=(2, 3))
row_length = np.sqrt(len(pos)).astype(int)
column_length = row_length

pos = np.round((pos - np.amin(pos)) / (np.amax(pos) - np.amin(pos)) * (row_length - 1)).astype(int)

velocities = np.empty(shape=(row_length * column_length, 2))

for coordinates in pos:
    index = coordinates[1] + coordinates[0] * row_length
    velocities[index] = vel[index]

t_max = 5  # simulation time in seconds
dt = 0.01 # step size
N = 2 ** 10  # Number of particles
D = 0.01  # diffusivity = 0.01 for final 

x_min = -1
x_max = 1
y_min = -1
y_max = 1

grid_size_x = 10
grid_size_y = 10

step_x = (abs(x_min)+abs(x_max))/grid_size_x # Step in grid (for x-direction)
step_y = (abs(y_min)+abs(y_max))/grid_size_y # Step in grid (for y-direction)

interval_x = np.arange(x_min, x_max+step_x, step_x)
interval_y = np.arange(y_min, y_max+step_y, step_y)
grid = np.zeros((grid_size_y,grid_size_x)) #Initialising grid array
 
#Blue particle initial postion
circle_r=0.3 #circle radius
#centre of circle coordinates
circle_x = (x_min+x_max)/2
circle_y = (y_min+y_max)/2

#Generate random point within a circle uniformly: https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
rand_rad = circle_r * np.sqrt(np.random.rand(N))
rand_angle = 2 * np.pi * np.random.rand(N)
x_blue = circle_x + rand_rad * np.cos(rand_angle)
y_blue = circle_y + rand_rad * np.sin(rand_angle)

#generate red dots
x_red = np.random.uniform(x_min, x_max, N)
y_red = np.random.uniform(y_min, y_max, N)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
color_gradient = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","purple","red"])

ax1.axis('scaled')
points_blue, = ax1.plot(x_blue, y_blue, 'bo', markersize= 0.5)
points_red, = ax1.plot(x_red, y_red,"ro", markersize = 0.5)
fig.suptitle("Frame: ", fontsize=20)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

ax2.axis('scaled')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

def get_velocities(x, y):
    x_coordinates = np.round((x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (row_length - 1)).astype(int)
    y_coordinates = np.round((y - np.amin(y)) / (np.amax(y) - np.amin(y)) * (row_length - 1)).astype(int)
    x_velocities = np.empty(shape=N)
    y_velocities = np.empty(shape=N)
    for i in range(N):
        velocity_index = y_coordinates[i] + x_coordinates[i] * row_length
        x_velocities[i] = velocities[velocity_index][0]
        y_velocities[i] = velocities[velocity_index][1]
    return x_velocities, y_velocities

def animate(time, x_blue, y_blue, points_blue, x_red, y_red, points_red, fig):
    v_x, v_y = get_velocities(x_blue, y_blue)
    x_blue += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrangian Diffusion
    y_blue += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrangian Diffusion
    x_red += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,))
    y_red += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,))
    for i in range(N):
        #Bounce particles off the boundary walls
        if x_blue[i] > x_max:
            x_blue[i] = 2 * x_max - x_blue[i]
        elif x_blue[i] < x_min:
            x_blue[i] = 2 * x_min - x_blue[i]
        if y_blue[i] > y_max:
            y_blue[i] = 2 * y_max - y_blue[i]
        elif y_blue[i] < y_min:
            y_blue[i] = 2 * y_min - y_blue[i]
        
        if x_red[i] > x_max:
            x_red[i] = 2 * x_max - x_red[i]
        elif x_red[i] < x_min:
            x_red[i] = 2 * x_min - x_red[i]
        if y_red[i] > y_max:
            y_red[i] = 2 * y_max - y_red[i]
        elif y_red[i] < y_min:
            y_red[i] = 2 * y_min - y_red[i]

    
    fig.suptitle("Frame: " + str(time), fontsize=20)
    points_blue.set_data(x_blue, y_blue)
    points_red.set_data(x_red, y_red)
    
    for z in range(0,grid_size_y):
        for j in range(0,grid_size_x):
            blue_indices = np.where(np.logical_and(x_blue>=interval_x[j], x_blue<=interval_x[j+1]))
            red_indices = np.where(np.logical_and(x_red>=interval_x[j], x_red<=interval_x[j+1]))
        
            for s in blue_indices:
                y_blue_intermediate = y_blue[blue_indices]
                
            for t in red_indices:
                y_red_intermediate = y_red[red_indices]
            
            y_blue_indices = np.where(np.logical_and(y_blue_intermediate>=interval_y[z], y_blue_intermediate<=interval_y[z+1]))
            y_red_indices = np.where(np.logical_and(y_red_intermediate>=interval_y[z], y_red_intermediate<=interval_y[z+1]))  
            blue_count = np.count_nonzero(y_blue_indices)
            red_count = np.count_nonzero(y_red_indices)
            both_count = blue_count+red_count
            
            if blue_count == 0:
                grid[(j,z)] = 1
                continue
            
            if red_count == 0:
                grid[(j,z)] = -1
                continue
        
            grid[(j,z)] = (-1*blue_count+red_count)/both_count

    conc_map = ax2.imshow(grid, interpolation='nearest', cmap= color_gradient ,extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(conc_map, cax=cax, orientation='vertical')
    
    return points_blue, points_red, conc_map

anim = animation.FuncAnimation(fig, animate, fargs=(x_blue, y_blue, points_blue, x_red, y_red, points_red, fig), frames=101, repeat=False, interval=1)
divider = make_axes_locatable(ax2) #This makes the color bar not stick out of the plot
cax = divider.append_axes('right', size='5%', pad=0.1) # This follows from the line above


ax1.title.set_text('Particles')
ax2.title.set_text('Concentration Grid')

plt.plot() 

