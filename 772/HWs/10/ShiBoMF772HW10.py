#%% Q3
import numpy as np
from math import sin, pi, exp
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
from math import floor
plt.style.use("seaborn")

lam = 0.05
M = 6
rho = 0.4
RR = 0.5
r = 0
LGD = 0.2
T = 5 
N = 50000
TE = 1

def s(T,rho,lam,RR,N,r,accrual):
    mu = np.zeros(5)
    sig = np.zeros((5,5)) + rho +  np.eye(5) * (1-rho)
    
    x1,x2,x3,x4,x5 = np.random.multivariate_normal(mu,sig,N).T
    u1 = norm.cdf(x1)
    u2 = norm.cdf(x2)
    u3 = norm.cdf(x3)
    u4 = norm.cdf(x4)
    u5 = norm.cdf(x5)
    
    t1 = -np.log(1-u1)/lam
    t2 = -np.log(1-u2)/lam
    t3 = -np.log(1-u3)/lam
    t4 = -np.log(1-u4)/lam
    t5 = -np.log(1-u5)/lam
    
    premium = np.zeros(N)
    deft = np.zeros(N)
    
    for i in range(N):
        tau = min(t1[i],t2[i],t3[i],t4[i],t5[i])
        if tau > T: # no default
            premium[i] = 5
        else:
            deft[i] = 1
            if accrual == 1:
                premium[i] = tau
            else:
                premium[i] = floor(tau)
        
    vp = premium.mean()
    vd = deft.mean()*LGD*(1-RR)
    
    s = vd/vp
    return s
spread = s(T,rho,lam,RR,N,r,0)
print("Spread: ",spread)

Rhoza_galeeva = []
delta_list = [i/1000 for i in range(1,100)]
for delta in delta_list:
    s1 = s(T,rho + rho * delta,lam,RR,N,r)
    s2 = s(T,rho - rho * delta,lam,RR,N,r)
    Rhoza_galeeva.append((s1-s2) / delta)
print("Sentivity dS/dR:",Rhoza_galeeva[-1])

plt.figure(dpi = 120)
plt.plot(delta_list, Rhoza_galeeva)
plt.xlabel("Delta")
plt.ylabel("Sensitivity")
plt.grid(True)
plt.show()
