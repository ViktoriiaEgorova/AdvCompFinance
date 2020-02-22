from math import *
import numpy as np 

def stats(array):
    N = array.size
    y = np.array(array, copy=True)
    x_tot = np.sum(y)
    m = x_tot/N

    for n in range( y .size ):
        y[n] -= m

    var = np.dot( y , y )
    sgma = var/N
    return m, var/N
# ------------------------------------------------

