# Linear density
# f(x) = x
import random
import math
import numpy as np


def lin_distr(a, b):
    print('Case 1: (a, b) = ('+ str(a) + ', ' + str(b)+')')
    #generating an array N
    N = np.arange(6, 19)
    for i in range(len(N)):
        N[i] = 2 ** N[i]
    print(N)

    # computing the normalization coefficient
    Z = 2.0/(b**2-a**2)
    # generating uniform RV using "random" library
    U = random.uniform(a, b)
    # getting variable X by solving equation
    X = math.sqrt(3*U+1)
    # computing mean m = E[X]
    m = Z * (b**3-a**3)/3.0
    # computing variance sigma squared and standard deviation
    sigma2 = Z * (b**4-a**4)/4.0 - m**2
    sigma = math.sqrt(sigma2)/N
    #print("%-7s%12s%12s%12s%12s%12s" % ('Z', 'U', 'X', 'm = E[X]', 'sigma^2 = E[X^2]-m^2', 'sigma'))
    print("%-7s%12s%12s%12s%12s%12s" % ('Z', 'U', 'X', 'm', 'sigma^2', 'sigma'))
    print("%-7f%12f%12f%12f%12f%12f" % (Z, U, X, m, sigma2, sigma ))





print('Linear density')
lin_distr(1, 2)