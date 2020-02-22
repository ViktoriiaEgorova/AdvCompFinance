#!/usr/bin/env python3
from math import *
import numpy as np 
import matplotlib.pyplot as plt
from stats import stats

def uniform_ab(Obj, N, a, b):
    array = Obj.uniform( low = 0.0, high = 1.0, size=N)
    for n in range ( array.size):
        array[n] = sqrt( (b**2 - a**2)*array[n] + a**2)

    return array
# -----------------------------------------------------

def f( a, b, x):
    g2 = b**2 - a**2
    y  = (2/g2)*x
    return y
#-------------------

if __name__ == "__main__":

    Obj = np.random.RandomState()
    # initialize RV generator
    # in order to have the same sequence (otherwise, every time sequence is different)
    Obj.seed(92823)

    #Ns = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    Ns = [6, 8, 10, 12, 14, 16, 18]
    bounds = [ (0, 1), (1, 2), ( 2, 5) ]


    ############################################################
    # pareto
    ############################################################
    
    #Nu = [ 2.10, 2.05, 1.95, 1.90]
    #a  = 1.2

    for a,b in bounds:
        M  = ( 2/(b**2 - a**2) ) * ( (b**3 - a**3 )/3)
        g2 = b**2 - a**2
        g3 = b**3 - a**3
        g4 = b**4 - a**4
        S2 =  (2./g2) * (g4/4) - M*M
        print()
        print("Uniform Distribution [%3.1f,%3.1f]" %(a,b))
        print("@ E[ X ] = %8.4f,  E[( X - M)^2] = %8.4f" %(M, S2))

        print("%9s    %8s %10s -- %8s   %8s -- %s "
                %("N", "E[X]", "E[(X-m)^2]", "|M-E[X]|", "MC-err", "Op"))
        for n in Ns:
            N     = ( 1 << n )
            array = uniform_ab( Obj, N, a, b)
            m, s2 = stats( array )
            err   = sqrt(s2/N)
            Op    = ( abs(m-M) < 2.*err )
            print("%9d    %8.4f %10.4f -- %8.2e   %8.2e -- %s " 
                    %( N, m, s2, abs(m-M), err, Op))

            if n >= 18:
                Dx = .01
                x  = np.arange(a, b+Dx, Dx)
                y  = f(a, b, x)

                dx = np.arange(a, b+Dx, Dx)

                plt.hist(array, density=True, facecolor='g', bins=dx)
                plt.plot(x, y, color='r')
                plt.show()
