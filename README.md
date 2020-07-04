# AdvCompFinance
Advanced Computational Finance UNIBO 2020

Pricing of the Asian call and put options, four barrier options (such as Down-and-Out call, Up-and-Out put, Down-and-In put, Up-and-In call) and Double Barrier option. All the payoff should be performed with Variance Gamma and Heston model.


The general algorithm is the following:
1. Set the parameters (either default or given from the command line). In particular, the term structure is imported from the file "ir.py".
2. Generate asset price trajectories depending on the model up to time T (which is the initial parameter, equals 1 year by default according to the task) using the Monte Carlo approach.
3. Check the martingale property of the generated trajectories in order to know if the computations can be reliable.
4. Depending of the type of the option compute the payoffs needed.
5. Compute the price of needed options and the corresponding error.

Also there are two separate files ("Heston checking.py ", "VG checking.py ") for testing the martingale property and reliability of the Monte Carlo simulations for option pricing for different maturities using fixed interest rates for two different cases. This is done in order to check the results with benchmark results and to understand if the results of
computation could be considered as relevant.
