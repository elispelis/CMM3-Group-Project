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
D = 0.01  # diffusivity = 0.1 for final 
# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

#Blue particle initial postion
circle_r=x_max/4 #circle radius
#centre of circle coordinates
circle_x = 0
circle_y = 0

#Generate random point within a circle uniformly : https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
rand_rad = circle_r * np.sqrt(np.random.rand(N))
rand_angle = 2 * np.pi * np.random.rand(N)
x = circle_x + rand_rad * np.cos(rand_angle)
y = circle_y + rand_rad * np.sin(rand_angle)

x_i = np.copy(x)
y_i = np.copy(y)

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

def animate(time, x, y, points, fig):
    v_x, v_y = get_velocities(x, y)
    x += v_x * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrangian Diffusion
    y += v_y * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=(N,)) #Lagrangian Diffusion
    for i in range(N):
        #Bounce particles off the boundary walls
        if x[i] > x_max:
            x[i] = 2 * x_max - x[i]
        elif x[i] < x_min:
            x[i] = 2 * x_min - x[i]
        if y[i] > y_max:
            y[i] = 2 * y_max - y[i]
        elif y[i] < y_min:
            y[i] = 2 * y_min - y[i]

    fig.suptitle("Frame: " + str(time), fontsize=20)
    points.set_data(x, y)
    return points

#Call out the functions and plot the animation
fig = plt.figure()
ax = plt.axes()
ax.axis('scaled')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
points, = ax.plot(x, y, 'o', markersize= 0.5)
fig.suptitle("Frame: ", fontsize=20)

anim = animation.FuncAnimation(fig, animate, fargs=(x, y, points, fig), frames=400, repeat=False, interval=5)
plt.show()