# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:42:04 2021

@author: Samir Elsheikh

"""

import numpy as np #Used to manipulate arrays and perform matrix operations
import matplotlib.pyplot as plt #Used to plot the relevant plots
from scipy.optimize import curve_fit #Used to fit the errors to a curve

#Inputs
avg_n = 5 #Number of runs to average over
N_list = [2**12, 2**10, 2**15, 2**14] #List of number of particles for which to run the simulation
N_list.sort(reverse = True) #Sort from highest to lowest.

h = [0.05, 0.01] #List of time steps for which to run the simulation
h.sort() #Sort from lowest to highest

D = 0.1

# End of Inputs

rmse_v = [] #Empty array to store rmse values that will be averaged

class TaskB:
    
    def find_nearest(self, array, value): # A function that returns the index of the closest number to a certain value in a list
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    def square(self, list): # A function that squares each element in a list
        return [i ** 2 for i in list]
    
    def __init__(self, N, h, avg_n, D):
        
        #Details
        self.t_max = 0.2 # Simulation time in seconds
        self.dt = h  # Step Size
        self.D = D  # Diffsivity
        self.Nx = 64 #Euler Grid Size
        self.N = N #Number of particles for which to run the simulation
        self.avg_n = avg_n #Number of times for which to run the simulation for each configuration of time step and number of particles
        
        # Domain size
        self.x_min = -1 #Lower bound for x-values
        self.x_max = 1 #Upper bound for x-values
        
        for i in range(self.avg_n): #Looping the process of getting values in order to process the error as an average
            self.run_numb = i + 1 #Counter that is used to print the run/iteration number
            self.initial_values() #Function that is defined in the next section
            self.sort_values() #Function that is defined in the next section
            self.loop() #Function that is defined in the next section
            self.compute() #Function that is defined in the next section
            self.loop2() #Function that is defined in the next section
             
        
    def initial_values(self): #This function initialises the simulation with arrays that will be used  later on along with getting the reference solution 
        self.ones = np.ones(self.N)  #Array of ones for where function
        self.zeros = np.zeros(self.N)  #Array of zeros for where function
        self.ref = np.genfromtxt('reference_solution_1D.dat') #Reads the reference solution CAN THIS BE KEPT OUTSIDE OF THE LOOP!!!!!!!!!
        self.x = np.random.uniform(self.x_min, self.x_max, size=self.N)  #Initial x-positions
        self.phi = np.where(self.x <= 0, self.ones, self.zeros) #Give x-coordinates phi values
    def sort_values(self): #This function sorts the arrays made in the initial_values function 
        self.x, self.phi = zip(*sorted(zip(self.x,self.phi))) #Sorts x into ascending order
        self.x_phi = np.column_stack((self.x,self.phi)) #Stacks x against phi
    def loop(self): #This function manipulates the phi values by incorporating the Diffusion Equation
        for i in np.arange(0, self.t_max, self.dt):
            self.x_phi[:,0] += np.sqrt(2 * self.D * self.dt) * np.random.normal(0, 1, size=self.N) #Diffusion Calulation
            self.x_phi[:,0][self.x_phi[:,0] < self.x_min] = self.x_min  #Any values in x thats under x_min replace to x_min
            self.x_phi[:,0][self.x_phi[:,0] > self.x_max] = self.x_max #Any values in x thats over x_max replace to x_max
    def compute(self): #This function calculates processes the outputs of the loop function
        self.avrphi = np.array([])    
        self.x_phi = self.x_phi[self.x_phi[:,0].argsort()] #Sort new x_phi into ascending
        self.phi_splits = np.split(self.x_phi[:,1], self.Nx) #Split phi values so average can be taken
    def loop2(self): #This function generates the final array for concentration
        for i in range(self.Nx):
            average	= np.cumsum(self.phi_splits[i])[-1]/len(self.phi_splits[i]) #Calculate average at Nx points
            self.avrphi=np.append(self.avrphi, average)
        self.x_val = np.linspace(self.x_min,self.x_max, self.Nx) #Evenly distributes Nx points from the lower x-bound to the the upper x-bound
        self.ref_x_ind = [] #Empty array to carry indices of the relevant x-positions that are closest to those found in the ones calculated in the loop functions
        self.ref_y =[] #Emptry array to carry the corresponding concentrations for each of the indices
        for j in range(self.x_val.size): #Loop for Nx (essentially size of the x_val array)
            d = self.ref[:,0] #Reference concentrations
            a = self.x_val[j] #Value to determine what the closest reference value to is
            s = self.find_nearest(d, a) #Find nearest x-value from reference compared to the points at which we have phi
            self.ref_x_ind.append(s) #Add to list of x-value indices

        for c in range(self.Nx): #Loop for Nx
            z = self.ref[:,1][self.ref_x_ind[c]] #Gets the concentration from the reference solution for the relevant indices
            self.ref_y.append(z) #Adds to the empty array created earlier
        
        self.error_list = [] #Empty array to carry the absolute error at each of the Nx points
        
        for k in range(self.Nx): #For each of the Nx points, this takes the difference between the reference and computed concentration values and adds it to the empty array
            error = abs(self.ref_y[k]-self.avrphi[k]) #Error calculation
            self.error_list.append(error) #Add error to empty array created earlier
        
        self.square_error = self.square(self.error_list) # This follows from the calculation from RMSE, which requires that the errors are squared
        self.f = sum(self.square_error) # The squared errors are added
        self.rmse = np.sqrt(self.f/(self.f.size)) #The square root of the sum of the squared errors divided by the number of errors obtained, yielding RMSE
        rmse_v.append(self.rmse) #Append this value to the empty array created before the definition of the TaskB class (Very top of the code)
        
        print("For run ",self.run_numb,", with ",self.N, " particles and " , self.dt, " time step, the root mean square error is ",self.rmse)

def plot(ref, x_min, x_max, Nx, f_avrphis): #Function to plot the concentrations for each configuration of time step and number of particles
    x = np.linspace(x_min, x_max, Nx) #Define plot limits and x-grid quantity  
    fig, ax = plt.subplots(1) #Creates the first plot
    ax.set_prop_cycle('color',plt.cm.rainbow(np.linspace(0,1,len(N_list)*len(h)))) #Defines the colourmap that defines how the color of the plots varies as more plots are added
    for i in range(len(f_avrphis)): #f_avrphis is defined below as an array which carries ALL the coordinates for ALL the configurations of time step and number of particles
        phi_int = f_avrphis[i] #Gets the coordinates of the ith configuration
        for z in range(len(phi_int)):
            phi2_int = phi_int[z] #Gets the concentration of zth entry
            plt.plot(x, phi2_int, marker='.', label='dT = %f and N = %.0f' %(h[i], N_list[z])) #Plot our values and label them with time step and number of particles
    
    plt.plot(ref[:, 0], ref[:, 1], color='Black', label = 'Reference at t = 0.2 s', linewidth=4.0, alpha=0.4) #Plot reference solution (with thicker line but a little transparent)   
    plt.title('Concentration vs Position at t = 0.2')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.grid() #Adding a grid to improve readability
    plt.legend() #Adds a legend according to the labels and colours already assigned in the plt.plot command
    plt.show() #Show the plot

f_avrphis = [] #Empty array to carry the concentration values for ALL configurations



for i in h:
   avrphis = [] #Empty array to hold the concentration values for configurations of same time step 
   for j in N_list:
       avrphis.append(TaskB(j, i, avg_n, D).avrphi) #Adds the concentration values to the empty array just created
   f_avrphis.append(avrphis) #Adds the concentration values to the empty f_avrphis array for ALL time steps and ALL number of particles
    


plot(np.genfromtxt('reference_solution_1D.dat'), -1, 1, 64, f_avrphis) #Plots the reference solution along with f_avrphis

rmse_avg = [] #Empty array for average RMSE values (averaged over avg_n runs)

avg_holder = [] #Holds average RMSE values for each iteration (for all time steps)

for z in range(0,len(h)): #Repeats for all time steps
    for i in range(0, len(N_list)): #Repeats for all number of particles
        rmse_holder = [] #Holds RMSE values temporarily for each iteration
        for j in range(0, avg_n):
            counter = j + avg_n * i + z * avg_n * len(N_list) #Skips through rmse_v to add all concentration values for the same number of particles 
            rmse_holder.append(rmse_v[counter]) #Collects the values as described in the line above
        entry_average = sum(rmse_holder)/len(rmse_holder) #Average across avg_n values
        avg_holder.append(entry_average)
        print(rmse_holder, "Average at this number of particles: ",entry_average)







# PLOTTING ERROR VS NUMBER OF PARTICLES FOR CONSTANT STEP SIZE (*All step sizes are considered)# 



    
def fit_func(N, a, b): #Function to be fitted to
    return a*N**b

for j in range(len(h)):
    pl_rmse = [] #Array for rmse values that will be plotted
    for i in range(len(N_list)):
        pl_rmse.append(avg_holder[i + j * len(N_list)])
    
    params = curve_fit(fit_func, N_list, pl_rmse) # Adapted from https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting - Fits our data to the function defined above
    [a, b] = params[0]

    fit_RMSE = [] #Empty array to carry fitted RMSE function 

    for z in N_list: #Getting values for the fitted RMSE solution at the same number of particles
        fit_RMSE_int = fit_func(z, a, b) 
        fit_RMSE.append(fit_RMSE_int) 

    fig, ax = plt.subplots(1) #Open a new figure for the Error vs Number of Particles plot
    ax.set_xscale('log') #Set the x-axis to a logarithmic scale
    ax.set_yscale('log') #Set the y-axis to a logarithmic scale 
    plt.xlabel('Number of Particles')
    plt.ylabel('RMSE')
    plt.text(.96,.94,'RMSE = %.3f*N^(%.3f)' %(a, b), bbox={'facecolor':'w','pad':5}, ha="right", va="top", transform=plt.gca().transAxes ) #Printing the fitted function on the plot with correct formatting
    plt.plot(N_list,pl_rmse, marker='.', label = 'Averaged RMSE over %.0f runs' %avg_n) #Plotting calculated RMSE
    plt.plot(N_list,fit_RMSE, label = 'Fitted RMSE', linestyle='dashed') #Plotting fitted RMSE
    plt.title('RMSE vs Number of Particles at t = 0.2 for dT = %f' %h[j])
    plt.legend(loc = 'lower left') #Add a legend



# Code ends here







"""
This part was an attempt to plot the relationship between step size and error, but no such relationship was found.


# PLOTTING ERROR VS. STEP SIZE FOR CONSTANT NUMBER OF PARTICLES (*largest number of particles is ALWAYS used)# 













pl2_rmse = [] #Array for rmse values that will be plotted (This is for the second error plot which is RMSE vs. Step-Size)


for i in range(len(h)):
    pl2_rmse.append(avg_holder[i * len(N_list)])

print(pl2_rmse)
print(h)

params = curve_fit(fit_func, h, pl2_rmse) # Adapted from https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
[a, b] = params[0]

fit_RMSE_2 = []

for i in h: #Getting values for the fitted RMSE solution at the same number of particles
    fit_RMSE_int_2 = fit_func(i, a, b)
    fit_RMSE_2.append(fit_RMSE_int_2) 



fig, ax = plt.subplots(1)
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('Step Size')
plt.ylabel('RMSE')
plt.text(.04,.94,'RMSE = %.3f*N^(%.3f)' %(a, b), bbox={'facecolor':'w','pad':5}, ha="left", va="top", transform=plt.gca().transAxes )
plt.plot(h,pl2_rmse, marker='.', label = 'Averaged RMSE over %.0f runs' %avg_n) #Plotting calculated RMSE
plt.plot(h,fit_RMSE_2, label = 'Fitted RMSE', linestyle='dashed') #Plotting fitted RMSE
plt.title('RMSE vs Step Size at t = 0.2 for N = %.0f particles' %N_list[0])
plt.legend(loc = 'lower left')
"""