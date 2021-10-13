import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation as animation

nsample = 1000
T=1.
D = 0.1
particles = 3

time = np.linspace(0,T,nsample)

dT = time[1]-time[0]

dX = np.sqrt(2*dT*D) * np.random.randn(particles,nsample)
dY = np.sqrt(2*dT*D) * np.random.randn(particles,nsample)


X = np.cumsum(dX,axis = 1)
Y = np.cumsum(dY,axis = 1)


for i in range(particles):
    plt.plot(X[i,:],Y[i,:])
plt.show()