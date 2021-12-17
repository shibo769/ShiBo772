# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:09:43 2021

@author: boshi
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
######################################
''' initial setup '''
nsim = 50000
Lambda = 0.04
p = [1, 0.5, 2]

def simulation(Lambda, p, nsim):
    '''
    Parameters
    ----------
    Lambda : Float
        Intensity of exponential or Weibull distribution.
    p : List
        Stores different p for Weibull distribution
    nsim : Int
        # of simulations wanted.

    Returns
    -------
    T1 : Array
        simulation results for p = 1 case
    T2 : Array
        simulation results for p = 0.5 case
    T3 : Array
        simulation results for p = 2 case
    '''
    T1 = np.zeros(nsim)
    T2 = np.zeros(nsim)
    T3 = np.zeros(nsim)
    
    u = np.random.uniform(0,1,nsim)
    
    for j in range(nsim):
        T1[j] = np.power(-1/(Lambda**p[0]) * np.log(1-u[j]), 1/p[0])
        T2[j] = np.power(-1/(Lambda**p[1]) * np.log(1-u[j]), 1/p[1])
        T3[j] = np.power(-1/(Lambda**p[2]) * np.log(1-u[j]), 1/p[2])
    
    return T1, T2, T3
###################################### (a)
'''Get simulation results '''
T1, T2, T3 = simulation(Lambda, p, nsim)

###################################### (b)
''' Plot histogram '''

sns.distplot(T1, color="b", bins = 30)
plt.xlabel("T")
plt.ylabel("Frequency")
plt.title("For Hazard Function with p = 1")
plt.show()

sns.distplot(T2, color="b", bins = 30)
plt.xlabel("T")
plt.ylabel("Frequency")
plt.title("For Hazard Function with p = 0.5")
plt.show()

sns.distplot(T3, color="b", bins = 30)
plt.xlabel("T")
plt.ylabel("Frequency")
plt.title("For Hazard Function with p = 2")
plt.show()
###################################### (c)
''' Mean and Variance '''

print("The mean value for p = 1 case is: " + str(np.mean(T1)))
print("The mean value for p = 0.5 case is: " + str(np.mean(T2)))
print("The mean value for p = 2 case is: " + str(np.mean(T3)))
print("The std for p = 1 case is: " + str(np.std(T1)))
print("The std for p = 0.5 case is: " + str(np.std(T2)))
print("The std for p = 2 case is: " + str(np.std(T3)))
###################################### (d)
mean1 = 1 / Lambda * gamma(1 + 1 / p[0])
mean2 = 1 / Lambda * gamma(1 + 1 / p[1])
mean3 = 1 / Lambda * gamma(1 + 1 / p[2])

var1 = np.sqrt(1 / Lambda**2 * (gamma(1+2/p[0]) - (gamma(1+1/p[0]))**2))
var2 = np.sqrt(1 / Lambda**2 * (gamma(1+2/p[1]) - (gamma(1+1/p[1]))**2))
var3 = np.sqrt(1 / Lambda**2 * (gamma(1+2/p[2]) - (gamma(1+1/p[2]))**2))

print("The theoretical mean for p = 1 is : ", mean1)
print("The theoretical mean for p = 0.5 is : ", mean2)
print("The theoretical mean for p = 2 is : ", mean3)

print("The theoretical std for p = 1 is : ", var1)
print("The theoretical std for p = 0.5 is : ", var2)
print("The theoretical std for p = 2 is : ", var3)

### continue
"""Error Analysis"""
n = [i for i in range(1000,50000,1000)]

error = []
for i in n:
    a,b,c = simulation(Lambda, [1,1,1], i)
    error.append(abs(np.mean(a) - mean1))

plt.figure(dpi = 120)
plt.plot(n,error,label = "abs error")
plt.legend()
plt.grid(True)
plt.xlabel("# of simulations")
plt.ylabel("Absoluate Error")
plt.title("Absoluate Error for (p = 1) Plot when n -> from 1000 to 50000")
