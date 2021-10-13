import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation as animation

nsample = 10
T=1.
D = 0.1
particles = 1

X = np.zeros((particles,nsample))
Y = np.zeros((particles,nsample))

time = np.linspace(0,T,nsample)

dT = time[1]-time[0]

dX = np.sqrt(2*dT*D) * np.random.randn(particles,nsample)
dY = np.sqrt(2*dT*D) * np.random.randn(particles,nsample)

print(X)

for i in range(nsample):
    X[i+1] = X[i] + dX[i]
    Y[i+1] = Y[i] + dY[i]


for i in range(particles):
    plt.plot(X[i,:],Y[i,:])
plt.show()