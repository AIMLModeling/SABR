import matplotlib.pyplot as plt

from pysabr import Hagan2002LognormalSABR
import numpy as np
sabrLognormal = Hagan2002LognormalSABR(f=2.5271/100, shift=3/100, t=10, beta=0.5)
strikes = np.array([-0.4729, 0.5271, 1.0271, 1.5271, 1.7771, 2.0271, 2.2771, 2.4021,
              2.5271, 2.6521, 2.7771, 3.0271, 3.2771, 3.5271, 4.0271, 4.5271,
              5.5271]) / 100
LogNormalVols = np.array([19.641923, 15.785344, 14.305103, 13.073869, 12.550007, 12.088721,
              11.691661, 11.517660, 11.360133, 11.219058, 11.094293, 10.892464,
              10.750834, 10.663653, 10.623862, 10.714479, 11.103755])
plt.xlabel('Strike') 
plt.ylabel('Volatility') 
plt.title("Volatility Smile")
plt.plot(strikes, LogNormalVols)
plt.show()
[alpha, rho1, volvol1] = sabrLognormal.fit(strikes, LogNormalVols)
print("Fitted  alpha, rho, volvol: ", [alpha, rho1, volvol1])



