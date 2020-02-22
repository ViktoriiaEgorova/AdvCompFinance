# LINEAR DISTRIBUTION

# PARAMETERS
bounds = [(0, 1), (1, 2), (2, 5)]
N = [2**n for n in [6, 8, 10, 12, 14, 16, 18]]

import numpy as np
from math import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # initialize RV generator
    Obj = np.random.RandomState()
    # in order to have the same sequence (otherwise, every time sequence is different)
    Obj.seed(92823)



    for a, b in bounds:

        print('( a, b ) = (', a, ', ', b, ')')

        # coefficient
        Z = 2.0 / (b**2 - a**2)
        # theoretical mean m = E[X] using only bounds
        M = Z * (b**3-a**3) / 3.0
        # theoretical variance Sigma^2 = E[(X-M)^2] using only bounds
        Sigma2 = Z * (b**4 - a**4) / 4.0 - M*M
        # print them
        print('Theoretical values:')
        print('M = E[X] = %8.4f, Sigma^2 = E[(X-M)^2] = %8.4f'  %(M, Sigma2))

        print('Values obtained from generated data:')
        print("%9s    %8s %10s -- %8s   %8s -- %s " % ("N", "E[X]", "E[(X-m)^2]", "|M-E[X]|", "MC-err", "Op"))

        for n in N:
            # generating an array of uniform RVs
            array = Obj.uniform(low=0.0, high=1.0, size=n)
            for k in range(array.size):
                array[k] = sqrt((b ** 2 - a ** 2) * array[k] + a ** 2)
            # we now have generated array of uniform RV
            m = np.mean(array)
            sigma2 = np.mean((array-m) * (array-m))
            err = sqrt(sigma2 / n)
            Op = (abs(m - M) < 2.0 * err)
            print("%9d    %8.4f %10.4f -- %8.2e   %8.2e -- %s " % (n, m, sigma2, abs(m - M), err, Op))

        Dx = .01
        x = np.arange(a, b + Dx, Dx)
        y = Z*x

        dx = np.arange(a, b + Dx, Dx)

        plt.hist(array, density=True, facecolor='g', bins=dx)
        plt.plot(x, y, color='r')
        plt.show()
