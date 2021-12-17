#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:31:36 2020

@author: shousakai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def f(x,y):
    return 1 / (1/x + 1/y - 1)

x = np.linspace(0.00000000001, 1, 1000)
y = np.linspace(0.00000000001, 1, 1000)

X, Y = np.meshgrid(x, y)

Z = f(X, Y)
plt.contour(X,Y,Z,colors="r")
plt.show()


M = 100000
np.random.seed(1)

lamba = 0.04

mean = np.zeros(10)

rho = np.arange(0.1,1,0.1)
prob = np.zeros(9)

for i in range(9):
    count = 0
    cov = rho[i] * np.ones((10, 10))
    for k in range(10):
        cov[k,k] = 1
    U = np.random.multivariate_normal(mean, cov, size = M)
    V = norm.cdf(U)
    tau = -np.log(1-V)/lamba
    for m in range(M):
        for j in range(10):
            if tau[m,j] <= 5:
                count += 1
                break
    prob[i] = count/M

plt.plot(rho, prob, "ro-")
plt.xlabel("correlation rho")
plt.ylabel("At least one default probability")
plt.show()            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    
    