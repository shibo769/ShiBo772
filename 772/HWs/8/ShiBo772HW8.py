import numpy as np
from scipy.stats import norm, gamma
from scipy.special import beta
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
plt.style.use("seaborn")
# from scipy.stats import beta 

p = 0.05
m = [100,1000]
rho = [0.1,0.5]
x = np.round(np.arange(0.05,0.95,0.1), 2)
N = 10000

def simu(p, m, rho):
    true_yn = []
    for i in x:
        y = []
        for j in range(N):
            z = np.random.normal(0,1)
            pz = norm.cdf((norm.ppf(p) - sqrt(rho) * z) / sqrt(1 - rho))
            u = np.random.uniform(0,1,m)
            X = np.where(u < pz, 1, 0)
            Nm = sum(X)
            if Nm / m < i:
                y.append(1)
        Y = len(y) / N
        true_yn.append(Y)
    return true_yn

def LPA(p, rho):
    prob = []
    for i in x:
        prob.append(norm.cdf(1 / sqrt(rho) * (sqrt(1-rho) * norm.ppf(i) - norm.ppf(p))))
    return prob

m100rho1 = simu(p, m[0], rho[0])
m100rho5 = simu(p, m[0], rho[1])
m1000rho1 = simu(p, m[1], rho[0])
m1000rho5 = simu(p, m[1], rho[1])
LPArho1 = LPA(p, rho[0])
LPArho5 = LPA(p, rho[1])

plt.figure(dpi = 100)
plt.plot(x, m100rho1, label = "Simulation (m = 100, rho = 0.1)")
plt.plot(x, m1000rho1,  label = "Simulation (m = 1000, rho = 0.1)")
plt.plot(x, LPArho1,  label = "LDA Approx rho = 0.1")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("CDF plot (rho = 0.1)", fontsize = 15)
plt.legend()

plt.figure(dpi = 100)
plt.plot(x, m100rho5, label = "Simulation (m = 100, rho = 0.5)")
plt.plot(x, m1000rho5,  label = "Simulation (m = 1000, rho = 0.5)")
plt.plot(x, LPArho5,  label = "LDA Approx rho = 0.5")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("CDF plot (rho = 0.5)", fontsize = 15)
plt.legend()
#%% Q3
p = 0.02
rho = 0.45

def f(z, p = 0.02, rho = 0.45):
    d = (norm.ppf(p) - sqrt(rho) * z)
    n = sqrt(1 - rho)
    temp = pow(norm.cdf( d / n ), 2) * norm.pdf(z)
    return temp

Epz2 = quad(f, -np.inf, np.inf)[0]

print("By numerical integration", (Epz2 - p**2) / (p * (1 - p)))
