#%% Functions
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sympy import *

def d1(V, d, T):
    d1 = (np.log(V/d) + (r + 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d1

def d2(V, d, T):
    d2 = (np.log(V/d) + (r - 0.5*sigma**2)*(T))/(sigma * np.sqrt(T))
    return d2

def Bm(V, d, T):
    Bm = d * np.exp(-r * T) * norm.cdf(d2(V,d,T)) + V * norm.cdf(-d1(V,d,T))
    return Bm

def y(V, d, T):
    y = np.log(d/Bm(V, d, T)) / T
    return y

def c_spread(V, d, T):
    s = -1/T * np.log(norm.cdf(d2(V,d,T)) + V/(d * np.exp(-r * T)) * norm.cdf(-d1(V,d,T)))
    return s

def P_D(V, d, T):
    P = 1- norm.cdf(d2(V, d, T))
    return P

def lev(V, d, T):
    L = D * np.exp(-r * T) / V
    return L

def reverse_spread(s_target, epsilon, a, b):
    '''
    s_target: your spread fixed
    epsilon: updating tolerance
    a : upper bound
    b : lower bound
    '''
    
    x = (a + b) / 2
    s = 0
    while abs(s - s_target) > epsilon:
        s = c_spread(V0, x, T)
        if s > s_target:
            b = x
        else:
            a = x
        x = (b + a) / 2
    
    return x
#%% Problem 1
V0 = 120
sigma = 0.25
T = 4
r = 0.05
D = np.array([40,100,180])

credit_spread = c_spread(V0, D, T)
print("The  credit spread s are: ", credit_spread)

P_D = P_D(V0,D,T)
df_P_D = pd.DataFrame({'D':D, 'Prob of default':P_D})
print("The probability of default at time T are:\n", df_P_D["Prob of default"])

d = np.array(list(range(125,150,1)))
cs = c_spread(V0, d, T)

print("The value of the debt for which credit spread s = 0.04 is ",\
      reverse_spread(0.04, 0.0001, 100,200))
#%% # Problem 2
sigma = 0.3
r = 0.05
T = np.array(range(1,100+1))
D_V0 = np.array([0.5, 1, 1.5])

for i in range(len(D_V0)):
    
    L_credit_spread = c_spread(1, D_V0[i], T)
    df = pd.DataFrame({'D/V0 %.3f credit spread'%D_V0[i]:L_credit_spread, 'T':T})
    df.set_index(['T'], inplace = True)
    df.plot()
#%% problem 3
sigma = 0.3
r = 0.05
T = 2
D_V0 = 1

# Q(VT<D)
Prob_Survival = norm.cdf(d2(1, D_V0, T))
Marginal = 1-Prob_Survival
# Q(R>50%)=Q(VT>0.5D)
Rgeq50 = norm.cdf(d2(1, D_V0*0.5, T))
# Q(R<=50%)
Rleq50 = 1 - Rgeq50
# Q(R>50%,VT<D)
joint = Marginal - Rleq50
# Q(R>50%|VT<D)
print("The probability of recovery > 50 conditional on default is: ", joint / Marginal)
