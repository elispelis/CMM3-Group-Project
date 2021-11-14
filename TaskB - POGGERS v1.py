# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:42:04 2021

@author: Samir Elsheikh


Q1: Do we average the rmse values across many runs? If so, how many runs?
Q2: How many time steps to consider? 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #Used to fit the errors to a curve

rmse_v = [] #Empty array to store rmse values that will be averaged
avg_n = 5 #Number of runs to average over


class TaskB:
    
    def find_nearest(self, array, value): # A function that returns the index of the closest number to a certain value in a list
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    def square(self, list): # A function that squares each element in a list
        return [i ** 2 for i in list]
    
    def __init__(self, N, h, avg_n):
        #Details
        self.t_max = 0.2  # simulation time in seconds
        self.dt = h  # step size
        self.D = 0.1  # diffsivity
        self.Nx = 64 #Euler grid size
        self.N = N
        self.avg_n = avg_n
        # Domain size
        self.x_min = -1
        self.x_max = 1
        
        for i in range(self.avg_n): #Looping the process of getting values in order to process the error as an average
            self.initial_values()    
            self.sort_values()
            self.loop()
            self.compute()
            self.loop2()
            
        
    def initial_values(self):
        self.ones = np.ones(self.N)  # Array of ones for where function
        self.zeros = np.zeros(self.N)  # Array of zeros for where function
        self.ref = np.genfromtxt('reference_solution_1D.dat')
        self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)  # initial x-positions
        self.phi = np.where(self.x <= 0, self.ones, self.zeros) #give x coordinates phi values
    def sort_values(self):
        self.x, self.phi = zip(*sorted(zip(self.x,self.phi))) #sorts x into ascending order
        self.x_phi = np.column_stack((self.x,self.phi)) #stack x against phi
    def loop(self):
        for i in np.arange(0, self.t_max, self.dt):
            self.x_phi[:,0] += np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=self.N) #diffusion calulation
            self.x_phi[:,0][self.x_phi[:,0] < self.x_min] = self.x_min  #any values in x thats under x_min replace to x_min
            self.x_phi[:,0][self.x_phi[:,0] > self.x_max] = self.x_max #any values in x thats over x_max replace to x_max
    def compute(self): 
        self.avrphi = np.array([])    
        self.x_phi = self.x_phi[self.x_phi[:,0].argsort()] #sort new x_phi into ascending
        self.phi_splits = np.split(self.x_phi[:,1], self.Nx) #split phi values so average can be taken
    def loop2(self):
        for i in range(self.Nx):
            average	= np.cumsum(self.phi_splits[i])[-1]/len(self.phi_splits[i]) #calculate averag at Nx points
            self.avrphi=np.append(self.avrphi, average)
        #SAMIR CHANGES
        self.x_val = np.linspace(self.x_min,self.x_max, self.Nx)
        self.ref_x_ind = []
        self.ref_y =[]
        for j in range(self.x_val.size):
            d = self.ref[:,0] #Reference information
            #print(d)
            a = self.x_val[j] #Value to determine what the closest reference value to is
            #print(a)
            s = self.find_nearest(d, a) #Find nearest x-value from reference compared to the points at which we have phi
            self.ref_x_ind.append(s) #Add to list of x-value indices

        for c in range(self.Nx):
            z = self.ref[:,1][self.ref_x_ind[c]]
            self.ref_y.append(z)
        
        self.error_list = []
        
        for k in range(self.Nx):
            error = abs(self.ref_y[k]-self.avrphi[k])
            self.error_list.append(error)
        
        self.square_error = self.square(self.error_list)
        self.f = sum(self.square_error)
        self.rmse = np.sqrt(self.f/(self.f.size))
        rmse_v.append(self.rmse)
        
        print("For ",self.N, " particles and " , self.dt, " time step, the root mean square error is ",self.rmse)

def plot(ref, x_min, x_max, Nx, f_avrphis):
    x = np.linspace(x_min, x_max, Nx) #Define plot limits and x-grid quantity  
    fig, ax = plt.subplots(1)
    ax.set_prop_cycle('color',plt.cm.rainbow(np.linspace(0,1,len(N_list)*len(h)))) #Defines the colourmap that the color of the plots will cycle through
    for i in range(len(f_avrphis)):
        phi_int = f_avrphis[i]
        for z in range(len(phi_int)):
            phi2_int = phi_int[z]
            plt.plot(x, phi2_int, marker='.', label='dT = %f and N = %.0f' %(h[i], N_list[z])) #Plot our values and label them with time step and number of particles
    
    plt.plot(ref[:, 0], ref[:, 1], color='Black', label = 'Reference', linewidth=4.0, alpha=0.4) #Plot reference solution (with thicker line but a little transparent)   
    plt.title('Concentration vs Position at t = 0.2')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.grid() #Adding a grid to improve readability
    plt.legend()
    plt.show() #Show the plot

N_list = [2**12, 2**15, 2**10, 2**11, 2**13, 2**14]
N_list.sort(reverse = True) #Sort from highest to lowest.

h = [0.05, 0.001]
h.sort() #Sort from lowest to highest

f_avrphis = []



for i in h:
   avrphis = []
   for j in N_list:
       avrphis.append(TaskB(j, i, avg_n).avrphi)
   f_avrphis.append(avrphis)    
    


plot(np.genfromtxt('reference_solution_1D.dat'), -1, 1, 64, f_avrphis)
print(rmse_v)

rmse_avg = []

avg_holder = [] #Holds average RMSE values for each iteration (for all time steps)

for z in range(0,len(h)):
    for i in range(0, len(N_list)):
        rmse_holder = [] #Holds RMSE values temporarily for each iteration
        for j in range(0, avg_n):
            counter = j + avg_n * i + z * avg_n * len(N_list)
            rmse_holder.append(rmse_v[counter])
        entry_average = sum(rmse_holder)/len(rmse_holder)
        avg_holder.append(entry_average)
        print(i)
        print(rmse_holder, "Average at this number of particles: ",entry_average)

print()
print("Average RMSE values list: ",avg_holder)

pl_rmse = [] #Array for rmse values that will be plotted




for i in range(len(N_list)):
    pl_rmse.append(avg_holder[i])
    
def fit_func(N, a, b):
    return a*N**b

params = curve_fit(fit_func, N_list, pl_rmse) # Adapted from https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
[a, b] = params[0]

fit_RMSE = []

for i in N_list: #Getting values for the fitted RMSE solution at the same number of particles
    fit_RMSE_int = fit_func(i, a, b)
    fit_RMSE.append(fit_RMSE_int) 



fig, ax = plt.subplots(1)
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('Number of Particles')
plt.ylabel('RMSE')
plt.text(.96,.94,'RMSE = %.3f*N^(%.3f)' %(a, b), bbox={'facecolor':'w','pad':5}, ha="right", va="top", transform=plt.gca().transAxes )
plt.plot(N_list,pl_rmse, marker='.', label = 'Averaged RMSE over %.0f runs' %avg_n) #Plotting calculated RMSE
plt.plot(N_list,fit_RMSE, label = 'Fitted RMSE', linestyle='dashed') #Plotting fitted RMSE
plt.title('RMSE vs Number of Particles at t = 0.2 for dT = %f' %h[0])
plt.legend(loc = 'lower left')