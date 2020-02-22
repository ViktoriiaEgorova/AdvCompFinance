# MARTINGAL PROPERTY CHECK

# PARAMETERS
T = 1.0 # 1 year
N = 12 # consider 12 monthly intervals
dt = T/N # intervals
J = 3 # number of trajectories
sigma = 0.3541
S0 = 0.987

import numpy as np
import random

if __name__ == "__main__":
    Obj = np.random.RandomState()
    Obj.seed(92823)
    for j in range(J):
        # considering new trajectory, fixing it
        Sj = np.zeros(N)
        M = np.zeros(N)
        V = np.zeros(N)
        # add the initial point ehich is the same for all trajectories
        Sj[0] = S0
        M[0] = S0
        # previous point is S0
        S_prev = S0
        # consider every interval (month by default)
        print('Trajectory ', j+1)
        # generating the random value
        rv = Obj.uniform(low=0.0, high=1.0, size=1)
        for n in range(1, N):
            # generating the random value
            rv = Obj.uniform(low=0.0, high=1.0, size=1)
            # computing the next point in the trajectory
            S_current = S_prev * np.exp((-(sigma**2)/2.0)*dt + sigma * np.sqrt(dt) * rv)
            # adding the new point to the trajectory
            Sj[n] = S_current
            # reassigning
            S_prev = S_current
            # studying some properties
            m = np.mean(Sj[:n+1])
            M[n] = m
            var = np.mean((Sj[:n+1]-m) * (Sj[:n+1]-m))
            V[n] = var


        # now we have our new trajectory built
        for i in range(len(Sj)):
            print('%8.7f %11.7f %14.7f' % (Sj[i], M[i], V[i]))






