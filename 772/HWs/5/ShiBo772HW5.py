#%% Problem 1
import pandas as pd
from scipy.stats import norm
# import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import minimize
import numpy as np

S =  45.52
T = 1.0
D = 97.0
r = 0.04
Sigma_S = 0.50722
error = 10**-4

V_initial = S + D * np.exp(-r * T)
Sigma_V0_initial = Sigma_S * S / V_initial

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
    F[1] = (1 / S * norm.cdf(d1(y, D, T, z)) * z * y) - Sigma_S
    
    return F

Gauss = np.array([V_initial, Sigma_V0_initial])

V, Sigma_V = fsolve(myfunc, Gauss)
new_S = V * norm.cdf(d1(V, D, T, Sigma_V)) - D * np.exp(-r * T) * norm.cdf(d2(V, D, T, Sigma_V))
new_Sigma_S = 1 / S * norm.cdf(d1(V, D, T, Sigma_V)) * Sigma_V * V

if abs(new_S - S) < error and abs(new_Sigma_S - Sigma_S) < error:
    print("(TRUE) The errors are less than 10^-4")
else:
    print("No")

print("The V should be ",V)
print("The Sigma_V should be", Sigma_V)


class SolveV:
    def __init__(self, sigma, ind):
        self.vol = sigma
        self.ind = ind
        self.T = 3

    def d1(self, V0):
        return (np.log(V0 / 100) + self.vol ** 2 / 2 * self.T) / (self.vol * np.sqrt(self.T))

    def d2(self, V0):
        return (np.log(V0 / 100) - self.vol ** 2 / 2 * self.T) / (self.vol * np.sqrt(self.T))

    def err(self, V0):
        return abs(data_df.iloc[self.ind, 1] - V0 * norm.cdf(self.d1(V0)) + 100 * norm.cdf(self.d2(V0)))

    def min_err(self):
        initial = data_df.iloc[self.ind, 1] + 100
        ans = minimize(self.err, initial)
        return ans.x

    def next(self):
        self.ind = self.ind + 1
        self.T = self.T - dt_list[self.ind]

    def reset(self, sigma1):
        self.vol = sigma1
        self.ind = 0
        self.T = 3
 
# problem 2
data_df = pd.read_excel('DataStockHW5.xlsx')
data_df['Date '] = pd.to_datetime(data_df['Date '], format='%Y/%m/%d')
data_df.rename(columns={'Stock  price': 'Stock'}, inplace=True)
data_df.dropna(axis=0, how='any', inplace=True)
dt_list = np.array(data_df['Date '] - data_df['Date '].shift(1))
dt_list = dt_list.astype('timedelta64[D]').astype(int) / 365
data_df['LogReturn'] = np.log(data_df['Stock'] / data_df['Stock'].shift(1)) / np.sqrt(dt_list)
data_df['V0'] = np.NaN
sigma0 = np.std(data_df.iloc[1:, 2])
print('sigma 0 =', sigma0)
solver0 = SolveV(sigma0, 0)
count = 0
pd.set_option('display.max_rows', None)

while True:
    data_df.iloc[0, 3] = solver0.min_err()
    for i in range(1, data_df.shape[0]):
        solver0.next()
        data_df.iloc[i, 3] = solver0.min_err()
    data_df['LogReturn'] = np.log(data_df['V0'] / data_df['V0'].shift(1)) / np.sqrt(dt_list)
    sigma_new = np.std(data_df.iloc[1:, 2])
    count += 1
    if abs(sigma_new - sigma0) <= 0.0001:
        print('Iteration', count, 'times')
        print('SigmaV =', sigma_new)
        break
    else:
        solver0.reset(sigma_new)
        sigma0 = sigma_new
        print('sigma', count, '=', sigma0)
data_df['DD'] = (data_df['V0'] - 100) / (data_df['V0'] * sigma_new)
print(data_df)
