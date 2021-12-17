# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:42:03 2020

@author: xiaoy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, fsolve, root
from scipy.stats import norm

# (2)
c=np.linspace(0,0.3,101)
def f(x):
    return abs(784*x**3+328*x**2+x-24)
def g(x):
    numer=np.sqrt(1/(4*x+1))-1/(2*x+1)
    denom=np.sqrt(1/(2*x+1))-1/(2*x+1)
    return abs(numer/denom-0.25)
plt.plot(c,f(c))
plt.show()
res=minimize(f,0.2)
print(res.x)
ans=minimize(g,0.2)
print('a',ans.x)
p_ave=np.sqrt(1/(2*ans.x[0]+1))
print('p average',p_ave)
print(np.sqrt(100*p_ave*(1-p_ave)))
v=100*p_ave*(1-p_ave)+100*99*(np.sqrt(1/(4*ans.x[0]+1))-p_ave**2)
#v=100*p_ave*(1-p_ave)+100*99*0.25*p_ave*(1-p_ave)
print(v)
print(np.sqrt(v))

# (3)
print(1000*0.01*0.99+1000*999*0.01*0.99*0.00901)
print(np.sqrt(1000*0.01*0.99+1000*999*0.01*0.99*0.00901))
print(1000000*0.01*0.99/99.009801)