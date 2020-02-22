# PARETO DISTRIBUTION

# PARAMETERS
#bounds = [(0, 1), (1, 2), (2, 5)]
N = [2**n for n in [6, 8, 10, 12, 14, 16, 18]]
Nu = [ 2.50, 2.05, 1.00]
a  = 1.2


import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f1(x, nu, a):
    return nu * a**nu * (1 / x**nu)

def f2(x, nu, a):
    return nu * a**nu * x**(1-nu)

if __name__ == "__main__":
    # initialize RV generator
    Obj = np.random.RandomState()
    # in order to have the same sequence (otherwise, every time sequence is different)
    Obj.seed(92823)



    for nu in Nu:

        print('Nu = ', nu)

        # coefficient
        Z = nu * a**nu
        # theoretical mean m = E[X] using only bounds
        M = quad(f1, a, np.inf, args=(nu, a))
        M = M[0]
        # theoretical variance Sigma^2 = E[(X-M)^2] using only bounds
        Sigma2 = quad(f2, a, np.inf, args=(nu, a))
        Sigma2 = Sigma2[0] - M**2
        # print them
        print('Theoretical values:')
        print('M = E[X] = %8.4f, Sigma^2 = E[(X-M)^2] = %8.4f'  %(M, Sigma2))

        print('Values obtained from generated data:')
        print("%9s    %8s %10s -- %8s   %8s -- %s " % ("N", "E[X]", "E[(X-m)^2]", "|M-E[X]|", "MC-err", "Op"))

        for n in N:
            # generating an array of uniform RVs
            array = Obj.uniform(low=0.0, high=1.0, size=n)
            for k in range(array.size):
                array[k] = a / (1 - array[k])**(1 / nu)
            # we now have generated array of uniform RV
            m = np.mean(array)
            sigma2 = np.mean((array-m) * (array-m))
            err = sqrt(sigma2 / n)
            Op = (abs(m - M) < 2.0 * err)
            print("%9d    %8.4f %10.4f -- %8.2e   %8.2e -- %s " % (n, m, sigma2, abs(m - M), err, Op))

        Dx = .01
        x = np.arange(a, 10*a + Dx, Dx)
        y = Z / (x**(1+nu))

        dx = np.arange(a, 10*a + Dx, Dx)

        plt.hist(array, density=True, facecolor='g', bins=dx)
        plt.plot(x, y, color='r')
        plt.show()
