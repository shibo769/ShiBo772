##% Q8
import numpy as np
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import factorial

def BSM_put(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    put_price = Nd2 * K * np.exp(-r * t) - S * Nd1
    return put_price
def BSM_call(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    call_price = -Nd2 * K * np.exp(-r * t) + S * Nd1
    return call_price
def DVA_CVA(F,K,sigma,T,r,N,collectoralA,collectoralB,RRa,RRb,col = False):
    if col == True:
        Call = BSM_call(F,K + collectoralB/N,sigma,T,0) * np.exp(-r * T)
        Put = BSM_put(F,K,sigma,T,0) * np.exp(-r * T)
        CVA = (1 - RRb) * PDb * Call * N
        DVA = (1 - RRa) * PDa * Put * N
    else:
        Call = BSM_call(F,K,sigma,T,0) * np.exp(-r * T)
        Put = BSM_put(F,K,sigma,T,0) * np.exp(-r * T)
        CVA = (1 - RRb) * PDb * Call * N
        DVA = (1 - RRa) * PDa * Put * N
    return CVA, DVA
N = 10000
T = 1
K = 1600
F = 1700
PDa = 0.015
PDb = 0.02
r = 0.03
sigma = 0.2
RRa = 0.4
RRb = 0.3
CVA,DVA = DVA_CVA(F,K,sigma,T,r,N,0,0,RRa,RRb, col = False)
print("Without collateral CVA and DVA: ",CVA,"|",DVA)
FV = np.exp(-r) * N * (F - K)
V = FV - CVA + DVA
print("V :",V)

#%% Q7
lam1 = 0.04
lam2 = 0.05
rho = 0.25
T = 5
r = 0.03/100
RR = 0.4
M = 2
N = 50000
def s(T,rho,lam1,lam2,RR,N,r):
    mu = np.zeros(2)
    cov = np.zeros((2,2)) + rho +  np.identity(2) * (1-rho)
    x1,x2 = np.random.multivariate_normal(mu,cov,N).T
    u1 = norm.cdf(x1)
    u2 = norm.cdf(x2)
    t1 = -np.log(1-u1)/lam1
    t2 = -np.log(1-u2)/lam2
    buy = np.zeros(N)
    sell = np.zeros(N)
    y = np.linspace(0,T,T+1)
    for i in range(N):
        t = min(t1[i],t2[i])
        for j in range(T):
            if t >= y[j+1]:
                buy[i] += np.exp(-r * y[j+1])
            if (t < y[j+1]) & (t > y[j]):
                buy[i] += (t - y[j]) * np.exp(-r * (t - y[j]))
        if t <= y[-1]:
            sell[i] = np.exp(-r * t) * (1 - RR) / 2
    s = sell.mean() / buy.mean() 
    return s
print(" spread s",s(T,rho,lam1,lam2,RR,N,r))  
#%% Q5
D = 70
T = 2
r = 0.04
sigma = 0.3
V0 = 100

def get_survivalP(V0, D, r, sigma, T, K1):
    d2 = (np.log(V0 / D) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d3 = (np.log(K1 ** 2 / (V0 * D)) + (r - sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    Nd2 = norm.cdf(d2)
    Nd3 = norm.cdf(d3)
    m = r / (0.5 * sigma ** 2)
    P = Nd2 - (V0 / K1) ** (- m + 1) * Nd3
    return P

f = lambda x: get_survivalP(V0, D, r, sigma, T, x) - 0.7

get_survivalP(V0, D, r, sigma, T, fsolve(f, 100))
D * np.exp(-r*T)

print("The Safety Coverant K1 is", fsolve(f, 100))
#%% Q1

pbar=0.02
rho=0.3
alpha=99.5/100
l=1
m=1

res=(norm.ppf(pbar)+norm.ppf(alpha)*rho**0.5)/(1-rho)**0.5
VaR=l * m * norm.cdf(res)
print("result:",VaR)