import matplotlib.pyplot as plt
import numpy as np

#import reference data
reference = "reference_solution_1D.dat"

#assign reference data to variables
x_ref = np.genfromtxt(reference, usecols = 0)
y_ref = np.genfromtxt(reference, usecols = 1)

#Details
t = 0.2  # simulation time in seconds
dt = 0.001  # step size
N = 2 ** 16  # Number of particles
D = 0.1  # diffusivity
Nx = 64
Ny = 1 #Euler grid size


# Domain size
x_min = -1
x_max = 1
y_min = -1
y_max = 1

for i in np.arange(0, (t+dt), dt):
    np.sqrt(2 * D * dt) * np.random.normal(0, 1, size=N) #Lagrange Diffusion and advection


plt.plot(x_ref,y_ref)
plt.show()
