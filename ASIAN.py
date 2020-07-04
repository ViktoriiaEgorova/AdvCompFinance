import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
import copy
from operator import itemgetter
import sys
from sys import argv
from sys import stdout as cout
from math import *

import ACF
import config


def Asian_GBM(S0, sigma, time_steps_T, J, strike, P_0_tM, P_0_T, BS_put, tM):

    print("ASIAN OPTION + GBM")

    Asset_GBM_trj_T = ACF.GBM(S0, 0, sigma, time_steps_T, J, strike)
    martingale = np.round(np.mean(Asset_GBM_trj_T, axis=1), 6)
    maturities = np.cumsum(time_steps_T)
    MC_error = np.sqrt((np.mean(Asset_GBM_trj_T ** 2, axis=1) - np.mean(Asset_GBM_trj_T, axis=1) ** 2) / J)

    layout1 = "{0:>15}{1:>15}{2:>15}"
    layout2 = "{0:>15}{1:>15}"
    layout3 = "{0:>15}{1:>15}{2:>15}{3:>15}"
    layout4 = "{0:>24}{1:>24}{2:>24}{3:>24}{4:>24}"
    print(len(martingale))
    print(len(maturities))

    print("Maringale check for the trajectories of the GBM used for the underlying")
    print()
    print(layout3.format('maturity', "Mean", "Abs error", "MC error"))
    print()
    for i, m in enumerate(martingale):
        if i != 0:
            print(layout3.format(maturities[i - 1], martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 np.round(3 * MC_error[i], 6)))
        else:
            print(layout3.format(0, martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 np.round(3 * MC_error[i], 6)))
    print()
    print()

    # ERROR ANALYSIS OF THE MONTE CARLO SIMULATION FOR PUT AND CALL OPTIONS

    MC_put_T = strike * P_0_T - Asset_GBM_trj_T[-1]
    MC_put_T[MC_put_T < 0] = 0
    MC_put_price_T = np.mean(MC_put_T)  # Put price obtained with the MC method

    square_payoff = np.mean(MC_put_T ** 2)

    error = np.sqrt((square_payoff - MC_put_price_T ** 2) / J)

    print(layout4.format("MC put price", "Theoretical price", 'number of simulations', "Absolute error", "Error"))
    print(layout4.format(np.round(MC_put_price_T, 6), BS_put, J, np.absolute(MC_put_price_T - BS_put), error))
    print()

    # ASIAN

    ST = np.sum(Asset_GBM_trj_T, 0) / len(time_steps_T)

    call = 0
    nu_call = 0
    put = 0
    nu_put = 0
    for j in range(len(ST)):
        temp_call = P_0_tM * np.maximum(ST[j] - strike, 0)
        temp_put = P_0_tM * np.maximum(strike - ST[j], 0)
        call = call + temp_call
        put = put + temp_put
        nu_call = nu_call + temp_call ** 2
        nu_put = nu_put + temp_put ** 2

    error_call = np.sqrt( (nu_call/J - (call/J)**2) / J )
    error_put = np.sqrt((nu_put/J - (put/J) ** 2) / J)
    call =  call / (len(ST))
    put =  put / (len(ST))


    print('Asian call option price (GBM) : ', round(call, 5), '  Error asian call (GBM) : ', round(error_call, 5) )
    print('Asian put option price (GBM)  : ', round(put, 5), '   Error asian put (GBM)  : ', round(error_put, 5))
    print(' ')




    # PLOTS

    plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_GBM_trj_T, axis=1))
    plt.xlabel("Time")
    plt.ylabel('Standar dev')
    plt.title("Evolution of the standard deviation (GBM)")
    plt.savefig("Evolution_of_the_standard_deviation (GBM).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]), maturities)), Asset_GBM_trj_T[:, 0:20])
    plt.xlabel("Time")
    plt.ylabel("Asset price")
    plt.title("Asset trajectories for the whole life of the contract (GBM)")
    plt.savefig("Asset_trajectories_for_the_whole_life_of_the_contract_(GBM).pdf")
    plt.show()

    # plt.hist(Asset_GBM_trj_T[-1], bins=1000, density=True)
    # plt.title("Density of the asset with GBM at tm = " + str(tM))
    # plt.savefig("Density_of_the_asset_with_GBM_at_tm = " + str(tM) + ".pdf")
    # plt.show()

def Asian_VG(S0, time_steps_T, N, strike, P_0_T, BS_put, P_0_tM, BS_Call):

    # Variance Gamma parameters
    eta = 0.1664
    theta = -.7678
    nu = 0.0622

    print("ASIAN OPTION + VG")

    Asset_VG_trj_T = ACF.VG(S0, eta, theta, nu, time_steps_T, N)

    maturities = np.cumsum(time_steps_T)

    layout1 = "{0:>15}{1:>15}{2:>15}"
    layout2 = "{0:>15}{1:>15}"
    layout3 = "{0:>15}{1:>15}{2:>15}{3:>15}"
    layout4 = "{0:>24}{1:>24}{2:>24}{3:>24}{4:>24}"

    martingale = np.round(np.mean(Asset_VG_trj_T, axis=1), 6)  # Computing the mean along each row

    MC_error = np.sqrt(
        (np.mean(Asset_VG_trj_T ** 2, axis=1) - np.mean(Asset_VG_trj_T, axis=1) ** 2) / N)  # MC error of the underlying

    print("Maringale check for the trajectories of the VG used for the underlying")
    print()
    print(layout3.format('maturity', "Mean", "Abs error", "MC error"))
    print()
    for i, m in enumerate(martingale):
        if i != 0:
            print(layout3.format(maturities[i - 1], martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 "%.6f" % (3 * np.round(MC_error[i], 6))))
        else:
            print(layout3.format(0, martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 "%.6f" % (3 * np.round(MC_error[i], 6))))
    print()
    print()

    # ERROR ANALYSIS OF THE MONTE CARLO SIMULATION FOR PUT AND CALL OPTIONS

    MC_put_T = strike * P_0_T - Asset_VG_trj_T[-1]
    MC_put_T[MC_put_T < 0] = 0
    MC_put_price_T = np.mean(MC_put_T)  # Put price obtained with the MC method

    square_payoff = np.mean(MC_put_T ** 2)  # MC error of the put option

    error = np.sqrt((square_payoff - MC_put_price_T ** 2) / N)

    # print(layout4.format("MC put price", "Theoretical price", 'N of simulations', "Absolute error", "Error"))
    # print(layout4.format(np.round(MC_put_price_T, 6), BS_put, N, np.absolute(MC_put_price_T - BS_put), error))
    # print()
    print(layout1.format("Theoretical call price", "Theoretical put price", 'N of simulations'))
    print(layout1.format(np.round(BS_Call, 7), np.round(BS_put, 7), N))
    print()



    # ASIAN

    ST = np.sum(Asset_VG_trj_T, 0) / len(time_steps_T)

    call = 0
    nu_call = 0
    put = 0
    nu_put = 0
    for j in range(len(ST)):
        temp_call = P_0_tM * np.maximum(ST[j] - strike, 0)
        temp_put = P_0_tM * np.maximum(strike - ST[j], 0)
        call = call + temp_call
        put = put + temp_put
        nu_call = nu_call + temp_call ** 2
        nu_put = nu_put + temp_put ** 2

    error_call = np.sqrt((nu_call / N - (call / N) ** 2) / N)
    error_put = np.sqrt((nu_put / N - (put / N) ** 2) / N)
    call = call / (len(ST))
    put = put / (len(ST))

    print('Asian call option price (VG) : ', round(call, 7), '  Error asian call (VG) : ', round(error_call, 7))
    print('Asian put option price (VG)  : ', round(put, 7), '   Error asian put (VG)  : ', round(error_put, 7))
    print(' ')

    # PLOTS

    plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_VG_trj_T, axis=1))
    plt.xlabel("Time")
    plt.ylabel('Standar dev')
    plt.title("Evolution of the standard deviation (VG)")
    plt.savefig("Evolution_of_the_standard_deviation(VG).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]), maturities)), MC_error)
    plt.xlabel("Time")
    plt.ylabel('MC_error')
    plt.title("Evolution of the MC_error (VG)")
    plt.savefig("Evolution_of_the_MC_error(VG).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]), maturities)), Asset_VG_trj_T[:, 0:20])
    plt.xlabel("Time")
    plt.ylabel("Asset price")
    plt.title("Asset trajectories for the whole life of the contract (VG)")
    plt.savefig("(VG)Asset_trajectories_for_the_whole_life_of_the_contract.pdf")
    plt.show()

    # plt.hist(Asset_VG_trj_tM[-1], bins=1000, density=True)
    # plt.title("Density of the asset with Variance Gamma at tm = " + str(tM))
    # plt.savefig("Density of the asset with Variance Gamma at tm = " + str(tM) + ".pdf")
    # plt.show()

def Asian_Heston(S0, time_steps_T, N, strike, P_0_T, BS_put, P_0_tM, BS_Call):

   # N = 2**20

    # Heston parameters
    sigma = .0635  # initial value for Heston
    k = .1153
    theta = .0240
    eta = .01
    rho = .2125

    print("ASIAN OPTION + HESTON")

    Asset_Heston_trj_T = ACF.heston(sigma, k, theta, eta, S0, 0, time_steps_T, rho, N)

    maturities = np.cumsum(time_steps_T)

    layout1 = "{0:>15}{1:>15}{2:>15}"
    layout2 = "{0:>15}{1:>15}"
    layout3 = "{0:>15}{1:>15}{2:>15}{3:>15}"
    layout4 = "{0:>24}{1:>24}{2:>24}{3:>24}{4:>24}"

    martingale = np.round(np.mean(Asset_Heston_trj_T, axis=1), 6)  # Computing the mean along each row

    MC_error = np.sqrt((np.mean(Asset_Heston_trj_T ** 2, axis=1) - np.mean(Asset_Heston_trj_T, axis=1) ** 2) / N)

    print("Maringale check for the trajectories of the GBM used for the underlying")
    print()
    print(layout3.format('maturity', "Mean", "Abs error", "MC error"))
    print()
    for i, m in enumerate(martingale):
        if i != 0:
            print(layout3.format(maturities[i - 1], martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 3 * np.round(MC_error[i], 7)))
        else:
            print(layout3.format(0, martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 3 * np.round(MC_error[i], 7)))
    print()
    print()

    # ERROR ANALYSIS OF THE MONTE CARLO SIMULATION FOR PUT AND CALL OPTIONS

    MC_put_T = strike * P_0_T - Asset_Heston_trj_T[-1]
    MC_put_T[MC_put_T < 0] = 0
    MC_put_price_T = np.mean(MC_put_T)  # Put price obtained with the MC method

    square_payoff = np.mean(MC_put_T ** 2)

    error = np.sqrt((square_payoff - MC_put_price_T ** 2) / N)

    # print(layout4.format("MC put price", "Theoretical price", 'N of simulations', "Absolute error", "Error"))
    # print(layout4.format(np.round(MC_put_price_T, 6), BS_put, N, np.absolute(MC_put_price_T - BS_put), error))
    # print()
    print(layout1.format("Theoretical call price", "Theoretical put price", 'N of simulations'))
    print(layout1.format(np.round(BS_Call, 7), np.round(BS_put, 7), N))
    print()

    # ASIAN

    ST = np.sum(Asset_Heston_trj_T, 0) / len(time_steps_T)


    call = 0
    nu_call = 0
    put = 0
    nu_put = 0
    for j in range(len(ST)):
        temp_call = P_0_tM * np.maximum(ST[j] - strike, 0)
        temp_put = P_0_tM * np.maximum(strike - ST[j], 0)
        call = call + temp_call
        put = put + temp_put
        nu_call = nu_call + temp_call ** 2
        nu_put = nu_put + temp_put ** 2

    error_call = np.sqrt((nu_call / N - (call / N) ** 2) / N)
    error_put = np.sqrt((nu_put / N - (put / N) ** 2) / N)
    call = call / (len(ST))
    put = put / (len(ST))

    print('Asian call option price (Heston) : ', round(call, 5), '  Error asian call (Heston) : ', round(error_call, 5))
    print('Asian put option price (Heston)  : ', round(put, 5), '   Error asian put (Heston)  : ', round(error_put, 5))
    print(' ')

    # PLOTS

    plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_Heston_trj_T, axis=1))
    plt.xlabel("Time")
    plt.ylabel('Standar dev')
    plt.title("Evolution of the standard deviation (Heston)")
    plt.savefig("Evolution_of_the_standard_deviation(Hest).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]), maturities)), MC_error)
    plt.xlabel("Time")
    plt.ylabel('MC_error')
    plt.title("Evolution of the MC_error (Heston)")
    plt.savefig("Evolution_of_the_MC_error(Heston).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]), maturities)), Asset_Heston_trj_T[:, 0:20])
    plt.xlabel("Time")
    plt.ylabel("Asset price")
    plt.title("Asset trajectories for the whole life of the contract (Hest)")
    plt.savefig("Asset_trajectories_for_the_whole_life_of_the_contract(Hest).pdf")
    plt.show()

    # plt.hist(Asset_Heston_trj_tM[-1], bins=1000, density=True)
    # plt.title("Density of the asset with Heston at tm = " + str(tM))
    # plt.savefig("(Hest)Density_of_the_asset_with_Heston_at_tm=" + str(tM) + ".pdf")
    # plt.show()




def usage():
    print("Usage: $> python3 ESAME [options]")
    print("Options parameters:")
    print("    %-24s: input_file: the input file holding interest rates")
    print("    %-24s: this output" % ("--help"))
    print("    %-24s: input data file (compulsory)" % ("-in"))
    print("    %-24s: method (compulsory) - VG / Heston" % ("-method"))
    print("    %-24s: initial value of the underlying (1 by default)" % ("-S0"))
    print("    %-24s: option strike (1.03 by default)" % ("-strike"))
    print("    %-24s: option maturity (1.13 by default)" % ("-T"))
    print("    %-24s: Length of the time steps (1/52 by default)" % ("-dt"))
    print("    %-24s: Number of simulations (2^J, J = 22 by default)" % ("-J"))


def run(args):
    # Default values

    J = 20  # Simulation of BS
    seed = 100

    np.random.seed(seed)

    # GBM and Black and Sholes parameters
    sigma = .24
    delta_t = 1 / 52
    # Option data
    tM = 1.0
    T = 1.0  # Option maturity
    tM = np.round(tM, 6)
    T = np.round(T, 6)
    strike = 1.03
    S0 = 1




    parms = config.get_input_parms(args)
    n_bins = 1000  # Number of bins for the histogram of the profit and loss function

    try:
        op = parms["help"]
        usage()
        return
    except KeyError:
        pass

    inpt = parms["in"]
    PAR = config.loadConfig(inpt)

    method = str(parms['method'])

    try:
        S0 = float(parms["S0"])
    except KeyError:
        pass
    try:
        strike = float(parms["strike"])
    except KeyError:
        pass
    try:
        T = float(parms["T"])
    except KeyError:
        pass

    try:
        sigma = float(parms["s"])
    except KeyError:
        pass
    try:
        delta_t = float(parms["dt"])
    except KeyError:
        pass
    try:
        J = int(parms["J"])
    except KeyError:
        pass
    try:
        n_bins = int(parms["bins"])
    except KeyError:
        pass
    try:
        seed = int(parms["seed"])
    except KeyError:
        pass

    J = 2 ** J
    np.random.seed(seed)

    ################################

    curve = PAR.curve
    final_TM = ACF.preparingStructure(curve, delta_t, T, tM)

    # COMPUTING THE PRICE OF THE PUT WITH BS
    index_T, index_tM, BS_put, BS_Call, P_0_T = ACF.put_call_price(final_TM, T, tM, S0, strike, sigma)

    P_0_tM = final_TM[index_tM, 1]  # Discount factor P(0,tM)

    # The Martingale property up to the maturity of the option
    # Selecting the time steps from 0 to T. They will be needed for verifing the martingale property
    time_steps_T = final_TM[:index_T + 1,4]

    if method == 'GBM':
        Asian_GBM(S0, sigma, time_steps_T, J, strike, P_0_tM, P_0_T, BS_put, tM)
    elif method == 'VG':
        Asian_VG(S0, time_steps_T, J, strike, P_0_T, BS_put, P_0_tM, BS_Call)
    elif method == 'Heston':
        Asian_Heston(S0, time_steps_T, J, strike, P_0_T, BS_put, P_0_tM, BS_Call)



if __name__ == "__main__":
    run(sys.argv)

