import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

lamb=0.04
rhos=np.arange(0.1,1,0.1)
trials=10000
results=[]
for i in range(len(rhos)):
    cov_matrix=np.array([[rhos[i]]*10 for _ in range(10)])
    res=0
    for trial in range(trials):
        for row in range(10):
            cov_matrix[row][row]=1
        normals=np.random.multivariate_normal([0]*10,cov_matrix)
        uniforms=norm.cdf(normals)
        default_times=-np.log(1-uniforms)/lamb
        if sum(default_times<5)>0:
            res+=1
    results.append(res/trials)
    
print(results)

plt.plot(rhos,results)
plt.xlabel("Rho")
plt.ylabel("Prob.")
plt.title("Prob. of First to Default Happens in 5 Years VS Rho")