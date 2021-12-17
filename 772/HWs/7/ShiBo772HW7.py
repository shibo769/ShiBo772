import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import norm

m = 10000
EDA = 100
LGD = 0.4
RR = 1 - LGD
p = 0.04
alpha = 0.99
D = [1000 ,500 ,200]

def Bin_VaR_ES(m, EDA, LGD, prob_default, alpha, D):
    '''
    D : Diversity Score
    '''
    RR = 1 - LGD
    L = EDA
    Loss = m * L / np.array(D)
    EL = np.array(D) * prob_default * Loss * (1-RR)
    
    num_of_default = binom.ppf(alpha, np.array(D), prob_default)
    
    VaR = num_of_default * Loss * (1 - RR)
    EC = VaR - EL
    if 1 - prob_default >= 1 - alpha:  
        ES = np.round((1 / (1-alpha)) * (VaR * (1 - prob_default)))
    else:
        ES = (1 / (1-alpha)) * (VaR * (1 - alpha))
    
    return VaR, EC, ES

print(Bin_VaR_ES(m, EDA, LGD, p, alpha, D))

