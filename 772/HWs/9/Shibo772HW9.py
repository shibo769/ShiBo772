import numpy as np
import matplotlib.pyplot as plt

def f(u,v,theta):
    return (u*v) / (1-theta*(1-u)*(1-v))

theta = [0,0.5,1]
u = np.linspace(0.00000000001, 1, 1000)
v = np.linspace(0.00000000001, 1, 1000)

U, V = np.meshgrid(u, v)

Z = f(U, V, theta[0])
plt.figure(dpi = 120)
plt.contour(U,V,Z)
plt.title("Theta = 0")
plt.xlabel("U")
plt.ylabel("V")
plt.show()

Z = f(U, V, theta[1])
plt.figure(dpi = 120)
plt.contour(U,V,Z)
plt.title("Theta = 0.5")
plt.xlabel("U")
plt.ylabel("V")
plt.show()

Z = f(U, V, theta[2])
plt.figure(dpi = 120)
plt.contour(U,V,Z)
plt.title("Theta = 1")
plt.xlabel("U")
plt.ylabel("V")
plt.show()