#%% 
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import factorial
from scipy.special import gamma

D = 105
S = 52.18
T = 2
r = 0.03
sigma_s = 0.5443
V_initial = S + D * np.exp(-r * T)
Sigma_V0_initial = sigma_s * S / V_initial

def d1(S, d, T, sigma):
    d1 = (np.log(S/d) + (r + 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d1

def d2(S, d, T, sigma):
    d2 = (np.log(S/d) + (r - 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d2

def myfunc(x):
    y = x[0]
    z = x[1]
    
    F = np.empty((2))
    F[0] = (y * norm.cdf(d1(y, D, T, z)) - D * np.exp(-r * T) * norm.cdf(d2(y, D, T, z))) - S
    F[1] = (1 / S * norm.cdf(d1(y, D, T, z)) * z * y) - sigma_s
    
    return F
Guess = np.array([V_initial, Sigma_V0_initial])
V, Sigma_V = fsolve(myfunc, Guess)
Sigma_B = 1 / D * norm.cdf(-d1(V, D, T, Sigma_V)) * Sigma_V * V
print("The V should be ",V)
print("The Sigma_V should be", Sigma_V)
print("The Sigma_B should be", Sigma_B)
expected_RR = V * np.exp(r * T) / D * (norm.cdf(-d1(V, D, T, Sigma_V)) / norm.cdf(-d2(V, D, T, Sigma_V)))
Prob_Default = 1-norm.cdf(d2(V, D, T, Sigma_V))
print("The expected RR is ", expected_RR)
print("The probability of default is ",Prob_Default)
### prob
# de = norm.cdf(-d2(V,D,T,0.6)) - norm.cdf(-d2(V,D,T,0.3))
# ne = norm.cdf(-d2(V,D,T,1))
#%%
V0 = 100
D = 80
T = 1
sigma = 0.15
Lambda = 0.4
nu = -0.04
delta = 0.3
r = 0.03
Lambda1 = Lambda * np.exp(nu + 0.5*delta**2)
K0 = Lambda1 / Lambda - 1
n = np.array([0,1,2,3,4,5,6,7])
sigma_n = np.sqrt(sigma**2 + n / T * delta**2)

def call_price_with_jump(V0, D, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, sigma_n, default, DEL):
    r_n = r - Lambda*K0 + (nu+delta**2 / 2) * n / T
    d1 = (np.log(V0/D) + (r_n + 0.5*sigma_n**2)*T) / np.sqrt(T) / sigma_n
    d2 = d1 - sigma_n * np.sqrt(T)
    factorials = []
    for i in n:
        factorials.append(factorial(i))
    weight = np.exp(-Lambda1*T) * ((Lambda1*T)**n) / np.array(factorials)
    BS_price = V0 * norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2)
    Call = sum(weight * BS_price)
    if default == True:
        Survival = sum(norm.cdf(d2) * weight)
        prob_default = 1 - Survival
        return Call, prob_default
    elif DEL == True:
        return norm.cdf(d1)
    else:
        return Call

### Implied Volatility
price = call_price_with_jump(V0, D, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, sigma_n, False,False)
f = lambda x: call_price_with_jump(V0, D, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, x, False,False) - price 
implied_v = fsolve(f, 0.5)
call_price_with_jump(V0, D, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, sigma_n, False, True)

prob_de = call_price_with_jump(V0, D, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, sigma_n, True,False)[1]
print("The call price with jump is", price)
print("The implied volatility is", implied_v)
print("Probability of default: ",prob_de)
#%%
### Volatility Smile
D_list = [i for i in range(10,210,1)]
V0_D = V0 / np.array(D_list)
p = []
vol = []

for i in D_list:
    p.append(call_price_with_jump(V0, i, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, sigma_n, False))

for i,j in zip(p,D_list):
    ff = lambda x: call_price_with_jump(V0, j, T, sigma, Lambda, nu, delta, r, Lambda1, K0, n, x, False) - i
    vol.append(fsolve(ff, 0.5))

plt.figure()
plt.plot(V0_D, vol)
plt.title("Volatility Smile")
plt.xlabel("V0/D")
plt.ylabel("Volatility")
plt.xlim([0.5,2])
plt.ylim([min(vol),0.35])

