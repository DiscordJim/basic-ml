import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator


X = np.linspace(0, 10, 100)

Y = np.random.randn(X.shape[0]) * 8

INTERVAL = (-2, 2, 5)

X = np.c_[np.ones(len(X)), X]

proj_x = []
proj_y = []
proj_z = []
for intercept in np.linspace(*INTERVAL):
    for slope in np.linspace(*INTERVAL):
        w = np.array([intercept, slope])
        loss = (1 / (2 * len(X))) * (np.square(np.dot(X, w) - Y).sum())
        proj_x.append(intercept)
        proj_y.append(slope)
        proj_z.append(loss)
        
proj_x = np.array(proj_x)
proj_y = np.array(proj_y)
proj_z = np.array(proj_z)
 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
# X = np.arange(-5, 5, 0.1)
# Y = np.arange(-5, 5, 0.1)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# Plot the surface.
surf = ax.plot_trisurf(proj_x, proj_y, proj_z, cmap=cm.autumn,
                       linewidth=0, antialiased=True)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()