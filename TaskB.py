# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:42:04 2021

@author: Samir Elsheikh


Q1: Do we average the rmse values across many runs? If so, how many runs?
Q2: How many time steps to consider? 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rmse_v = []

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
        self.initial_values()
        for i in range(self.avg_n):
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
    for i in range(len(f_avrphis)):
        phi_int = f_avrphis[i]
        
        if i == 0:
            for z in phi_int:
                plt.plot(x, z, marker='.', color='g') #Plot our values
        
        if i == 1:
            for z in phi_int:
                plt.plot(x, z, marker='.', color='c') #Plot our values
                
        if i == 2:
            for z in phi_int:
                plt.plot(x, z, marker='.', color='m') #Plot our values
                
        if i == 3:
            for z in phi_int:
                plt.plot(x, z, marker='.', color='y') #Plot our values
                
        if i == 4:
            for z in phi_int:
                plt.plot(x, z, marker='.', color='k') #Plot our values
    
    red_patch = mpatches.Patch(color='r', label="Reference")
    green_patch = mpatches.Patch(color='g', label="%s Particles" % N_list[0])
    cyan_patch = mpatches.Patch(color='c', label="%s Particles" % N_list[1])
    magenta_patch = mpatches.Patch(color='m', label="%s Particles" % N_list[2])
    
    plt.plot(ref[:, 0], ref[:, 1], color='r') #Plot reference solution       
    
    plt.legend(handles=[red_patch, green_patch, cyan_patch, magenta_patch])       
    plt.show() #Show the plot

N_list = [2**18,2**15, 2**12]
h = [0.01]

f_avrphis = []


for i in N_list:
   avrphis = []
   
   for j in h:
       avrphis.append(TaskB(i, j, 5).avrphi)   
   f_avrphis.append(avrphis)    
    


plot(np.genfromtxt('reference_solution_1D.dat'), -1, 1, 64, f_avrphis)
print(rmse_v)



#print("For 2**18", tw[0])
