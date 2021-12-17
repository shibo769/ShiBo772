#%% Problem 1 (a)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

D = 10
T = 2
r = 0.04
sigma = 0.2
V0 = 12.5

rho_list = np.array([i for i in np.arange(0.05 ,1.05, 0.05)])

K1_list = D * np.exp(-r*T) * rho_list

def d1(V, d, T):
    d1 = (np.log(V/d) + (r + 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d1

def d2(V, d, T):
    d2 = (np.log(V/d) + (r - 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d2

def get_survivalP(V0, D, r, sigma, T, K1):
    d2 = (np.log(V0 / D) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d3 = (np.log(K1 ** 2 / (V0 * D)) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    Nd2 = norm.cdf(d2)
    Nd3 = norm.cdf(d3)
    m = r / (0.5 * sigma ** 2)
    P = Nd2 - (V0 / K1) ** (- m + 1) * Nd3
    return P

def c_spread(V, d, T):
    s = -1/T * np.log(norm.cdf(d2(V,d,T)) + V/(d * np.exp(-r * T)) * norm.cdf(-d1(V,d,T)))
    return s

c_spread(V0, D, T)

defaultP_list = np.array([1 - get_survivalP(V0, D, r, sigma, T, i) for i in K1_list])
defaultP_list

plt.figure()
plt.title('Default P vs rho')
plt.plot(rho_list, defaultP_list)
plt.xlabel("rho")
plt.ylabel("Default P")
plt.show()
#%% Problem 1 (b)
spread = c_spread(V0, K1_list, T)

plt.figure()
plt.plot(spread, defaultP_list)
plt.ylabel("Default Probability")
plt.xlabel("Spread")
plt.title("Spread VS Default")
plt.show()
#%% Problem 1 (c)
# Set up the asset prices scenarios by Monte Carlo. Consider ρ = 0.9 and ρ = 0.0001
# Verify the analytical values for default probability by Monte Carlo simulations.
def simulate_V(V0, r, sigma, T, steps, N, K, D):
    h = T / steps
    dw = np.random.normal(
        0, np.sqrt(h), [N, steps])
    dV = r * h + sigma * dw
    c = np.ones([N])
    K_list = c * K
    D_list = c * D

    V = np.zeros([N, steps+1])
    V[:, 0] = V0
    for i in range(1, steps+1):
        V[:, i] = V[:, i-1] + V[:, i-1] * dV[:, i-1]
        c = (V[:, i] >= K_list) * c
        V[:, i] = V[:, i] * c
    c = (V[:, -1] >= D_list) * c
    V[:, -1] = V[:, -1] * c
    return V

rho = 0.9
N = 10000
steps = 10000
K1 = K1_list[-1] * rho
res = simulate_V(V0, r, sigma, T, N, steps, K1, D)
sum(res[:,-1]==0)/N

1 - get_survivalP(V0, D, r, sigma, T, K1)

# plt.figure(figsize=(10,6))
# plt.plot(pd.DataFrame(res).T)
#%%
rho = 0.0001
N = 10000
steps = 10000
K2 = D * np.exp(-r*T) * rho
res2 = simulate_V(V0, r, sigma, T, N, steps, K2, D)
sum(res2[:,-1]==0)/N

1 - get_survivalP(V0, D, r, sigma, T, K2)

# plt.figure(figsize=(10,6))
# plt.plot(pd.DataFrame(res2).T)