#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 01:05:47 2021

@author: dean, elis
"""

'''Simulation'''

#Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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
# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1
 
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
    return points_blue, points_red

#Call out the functions and plot the animation
fig = plt.figure()
ax = plt.axes()
ax.axis('scaled')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
points_blue, = ax.plot(x_blue, y_blue, 'bo', markersize= 0.5)
points_red, = ax.plot(x_red, y_red,"ro", markersize = 0.5)
fig.suptitle("Frame: ", fontsize=20)

anim = animation.FuncAnimation(fig, animate, fargs=(x_blue, y_blue, points_blue, x_red, y_red, points_red, fig), frames=10000, repeat=False, interval=5)
plt.plot()
#anim.save('test.gif')