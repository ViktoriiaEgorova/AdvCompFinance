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


def Barrier_GBM(S0, sigma, time_steps_T, J, strike, P_0_tM, P_0_T, BS_put, tM, lambda_min, lambda_max):

    print("BARRIER OPTIONS + GBM")

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

    # BARRIER

    ST = Asset_GBM_trj_T[-1][:]
    ST_min = np.amin(Asset_GBM_trj_T, 0)
    ST_max = np.amax(Asset_GBM_trj_T, 0)

    # a = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # np.where(a < 5, a, 10 * a)
    # array([0, 1, 2, 3, 4, 50, 60, 70, 80, 90])

    payoff_Cdo = np.where( ST_min > lambda_min, np.maximum(ST - strike, 0), 0 )
    payoff_Puo = np.where( ST_max < lambda_max, np.maximum(strike - ST, 0), 0 )
    payoff_Pdi = np.where( ST_min < lambda_min, np.maximum(strike - ST, 0), 0 )
    payoff_Cui = np.where( ST_max > lambda_max, np.maximum(ST - strike, 0), 0 )
    payoff_double = np.where( (lambda_min < ST_min) & (ST_max < lambda_max), ST, 0 )


    option_Cdo = P_0_tM * np.sum(payoff_Cdo) / J
    option_Puo = P_0_tM * np.sum(payoff_Puo) / J
    option_Pdi = P_0_tM * np.sum(payoff_Pdi) / J
    option_Cui = P_0_tM * np.sum(payoff_Cui) / J
    option_double = P_0_tM * np.sum(payoff_double) / J

    print('Down and Out call (GBM) : ', round(option_Cdo, 5))
    print('Up and Out put (GBM) : ', round(option_Puo, 5))
    print('Down and In put (GBM) : ', round(option_Pdi, 5))
    print('Up and In call (GBM) : ', round(option_Cui, 5))
    print('Double barrier option (GBM) : ', round(option_double, 5))



    #
    # print('Asian call option price (GBM) : ', round(call, 5))
    # print('Asian put option price (GBM) : ', round(put, 5))
    # print(' ')

    # # PLOTS
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_GBM_trj_T, axis=1))
    # plt.xlabel("Time")
    # plt.ylabel('Standar dev')
    # plt.title("Evolution of the standard deviation (GBM)")
    # plt.savefig("Evolution_of_the_standard_deviation (GBM).pdf")
    # plt.show()
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), Asset_GBM_trj_T[:, 0:20])
    # plt.xlabel("Time")
    # plt.ylabel("Asset price")
    # plt.title("Asset trajectories for the whole life of the contract (GBM)")
    # plt.savefig("Asset_trajectories_for_the_whole_life_of_the_contract_(GBM).pdf")
    # plt.show()
    #
    # # plt.hist(Asset_GBM_trj_T[-1], bins=1000, density=True)
    # # plt.title("Density of the asset with GBM at tm = " + str(tM))
    # # plt.savefig("Density_of_the_asset_with_GBM_at_tm = " + str(tM) + ".pdf")
    # # plt.show()

def Barrier_VG(S0, time_steps_T, N, strike, P_0_T, BS_put, P_0_tM, lambda_min, lambda_max, BS_Call):

    # Variance Gamma parameters
    eta = 0.1664
    theta = -.7678
    nu = 0.0622

    print("BARRIER OPTIONS + VG")

    Asset_VG_trj_T = ACF.VG(S0, eta, theta, nu, time_steps_T, N)

    maturities = np.cumsum(time_steps_T)

    layout1 = "{0:>24}{1:>24}{2:>24}"
    layout2 = "{0:>25}{1:>25}"
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
                                 "%.6f" % ( 3 * np.round(MC_error[i], 6))))
        else:
            print(layout3.format(0, martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 "%.6f" % ( 3 * np.round(MC_error[i], 6))))
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

    # BARRIER

    ST = Asset_VG_trj_T[-1][:]
    ST_min = np.amin(Asset_VG_trj_T, 0)
    ST_max = np.amax(Asset_VG_trj_T, 0)

    payoff_Cdo = np.where(ST_min > lambda_min, np.maximum(ST - strike, 0), 0)
    payoff_Puo = np.where(ST_max < lambda_max, np.maximum(strike - ST, 0), 0)
    payoff_Pdi = np.where(ST_min < lambda_min, np.maximum(strike - ST, 0), 0)
    payoff_Cui = np.where(ST_max > lambda_max, np.maximum(ST - strike, 0), 0)
    payoff_double = np.where((lambda_min < ST_min) & (ST_max < lambda_max), ST, 0)


    option_Cdo = P_0_tM * np.sum(payoff_Cdo) / len(payoff_Cdo)
    option_Puo = P_0_tM * np.sum(payoff_Puo) / len(payoff_Puo)
    option_Pdi = P_0_tM * np.sum(payoff_Pdi) / len(payoff_Pdi)
    option_Cui = P_0_tM * np.sum(payoff_Cui) / len(payoff_Cui)
    option_double = P_0_tM * np.sum(payoff_double) / len(payoff_double)


    nu_Cdo = np.sum(payoff_Cdo ** 2) / N
    error_Cdo = nu_Cdo - (np.sum(payoff_Cdo) / N) ** 2
    error_Cdo = np.sqrt(error_Cdo / N)
    nu_Puo = np.sum(payoff_Puo ** 2) / N
    error_Puo = nu_Puo - (np.sum(payoff_Puo) / N) ** 2
    error_Puo = np.sqrt(error_Puo / N)
    nu_Pdi = np.sum(payoff_Pdi ** 2) / N
    error_Pdi = nu_Pdi - (np.sum(payoff_Pdi) / N) ** 2
    error_Pdi = np.sqrt(error_Pdi / N)
    nu_Cui = np.sum(payoff_Cui ** 2) / N
    error_Cui = nu_Cui - (np.sum(payoff_Cui) / N) ** 2
    error_Cui = np.sqrt(error_Cui / N)
    nu_double = np.sum(payoff_double ** 2) / N
    error_double = nu_double - (np.sum(payoff_double) / N) ** 2
    error_double = np.sqrt(error_double / N)

    print('Down and Out call     (VG) : ', round(option_Cdo, 7),    '  Error Down and Out call     (VG) : ', round(error_Cdo, 7))
    print('Up   and Out put      (VG) : ', round(option_Puo, 7),    '  Error Up   and Out put      (VG) : ', round(error_Puo, 7))
    print('Down and In  put      (VG) : ', round(option_Pdi, 7),    '  Error Down and In  put      (VG) : ', round(error_Pdi, 7))
    print('Up   and In  call     (VG) : ', round(option_Cui, 7),    '  Error Up   and In  call     (VG) : ', round(error_Cui, 7))
    print('Double barrier option (VG) : ', round(option_double, 7), '  Error Double barrier option (VG) : ', round(error_double, 7))

    # # ANALYTICAL
    # H = lambda_min
    # K = strike
    # la = (r + (sigma**2) / 2) / (sigma**2)
    # if H<=K:
    #     y = np.log(H*H / (S0*K)) / (sigma * np.sqrt(T)) + H * sigma * np.sqrt(T)
    #     c_di = S0 * np.power(H/S0, 2*la) * norm.cdf(y) - K * np.exp(-r*T) * np.power(H/S0, 2*la-2) * norm.cdf(y - sigma*np.sqrt(T))
    #     c_do = BS_Call - c_di
    # else:
    #     x1 = np.log(S0/H) / (sigma * np.sqrt(T)) + H * sigma * np.sqrt(T)
    #     y1 = np.log(H/S0) / (sigma * np.sqrt(T)) + H * sigma * np.sqrt(T)
    #     c_do = S0 * norm.cdf(x1) - K * np.exp(-r*T) * norm.cdf(x1 - sigma*np.sqrt(T)) -S0 * np.power(H/S0, 2*la) *  norm.cdf(y1) + K * np.exp(-r*T)* np.power(H/S0, 2*la-2) * norm.cdf(y1 - sigma*np.sqrt(T))

    # print('c_do :  ', c_do)
    # print(' ')
    #
    # # SPURIOUS PARITY
    # temp = np.sum(ST) / len(ST) - strike * P_0_tM
    # print("DOC - DIP = ", option_Cdo - option_Pdi, ' = ', temp)
    # print("UIC - UOP = ", option_Cui - option_Puo, ' = ', temp)

    # # PLOTS
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_VG_trj_T, axis=1))
    # plt.xlabel("Time")
    # plt.ylabel('Standar dev')
    # plt.title("Evolution of the standard deviation (VG)")
    # plt.savefig("Evolution_of_the_standard_deviation(VG).pdf")
    # plt.show()
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), Asset_VG_trj_T[:, 0:20])
    # plt.xlabel("Time")
    # plt.ylabel("Asset price")
    # plt.title("Asset trajectories for the whole life of the contract (VG)")
    # plt.savefig("(VG)Asset_trajectories_for_the_whole_life_of_the_contract.pdf")
    # plt.show()
    #
    # # plt.hist(Asset_VG_trj_tM[-1], bins=1000, density=True)
    # # plt.title("Density of the asset with Variance Gamma at tm = " + str(tM))
    # # plt.savefig("Density of the asset with Variance Gamma at tm = " + str(tM) + ".pdf")
    # # plt.show()

def Barrier_Heston(S0, time_steps_T, N, strike, P_0_T, BS_put, P_0_tM, lambda_min, lambda_max, BS_Call):

   # N = 2**20

    # Heston parameters
    sigma = .0635  # initial value for Heston
    k = .1153
    theta = .0240
    eta = .01
    rho = .2125

    print("BARRIER OPTIONS + HESTON")

    Asset_Heston_trj_T = ACF.heston(sigma, k, theta, eta, S0, 0, time_steps_T, rho, N)

    maturities = np.cumsum(time_steps_T)


    layout1 = "{0:>24}{1:>24}{2:>24}"
    layout2 = "{0:>25}{1:>25}"
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
                                 "%.6f" % ( 3 * np.round(MC_error[i], 6))))
        else:
            print(layout3.format(0, martingale[i], np.absolute(np.round(S0 - martingale[i], 6)),
                                 "%.6f" % ( 3 * np.round(MC_error[i], 6))))
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

   # BARRIER


    ST = Asset_Heston_trj_T[-1][:]
    ST_min = np.amin(Asset_Heston_trj_T, 0)
    ST_max = np.amax(Asset_Heston_trj_T, 0)

    # a = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # np.where(a < 5, a, 10 * a)
    # array([0, 1, 2, 3, 4, 50, 60, 70, 80, 90])

    payoff_Cdo = np.where(ST_min > lambda_min, np.maximum(ST - strike, 0), 0)
    payoff_Puo = np.where(ST_max < lambda_max, np.maximum(strike - ST, 0), 0)
    payoff_Pdi = np.where(ST_min < lambda_min, np.maximum(strike - ST, 0), 0)
    payoff_Cui = np.where(ST_max > lambda_max, np.maximum(ST - strike, 0), 0)
    payoff_double = np.where((lambda_min < ST_min) & (ST_max < lambda_max), ST, 0)

    option_Cdo = P_0_tM * np.sum(payoff_Cdo) / N
    option_Puo = P_0_tM * np.sum(payoff_Puo) / N
    option_Pdi = P_0_tM * np.sum(payoff_Pdi) / N
    option_Cui = P_0_tM * np.sum(payoff_Cui) / N
    option_double = P_0_tM * np.sum(payoff_double) / N


    nu_Cdo = np.sum(payoff_Cdo **2) / N
    error_Cdo = nu_Cdo - (np.sum(payoff_Cdo) / N) **2
    error_Cdo = np.sqrt(error_Cdo/N)
    nu_Puo = np.sum(payoff_Puo ** 2) / N
    error_Puo = nu_Puo - (np.sum(payoff_Puo) / N) ** 2
    error_Puo = np.sqrt(error_Puo / N)
    nu_Pdi = np.sum(payoff_Pdi ** 2) / N
    error_Pdi = nu_Pdi - (np.sum(payoff_Pdi) / N) ** 2
    error_Pdi = np.sqrt(error_Pdi / N)
    nu_Cui = np.sum(payoff_Cui ** 2) / N
    error_Cui = nu_Cui - (np.sum(payoff_Cui) / N) ** 2
    error_Cui = np.sqrt(error_Cui / N)
    nu_double = np.sum(payoff_double ** 2) / N
    error_double = nu_double - (np.sum(payoff_double) / N) ** 2
    error_double = np.sqrt(error_double / N)

    print('Down and Out call     (Heston) : ', round(option_Cdo, 7),    '  Error Down and Out call     (Heston) : ', round(error_Cdo, 7))
    print('Up   and Out put      (Heston) : ', round(option_Puo, 7),    '  Error Up   and Out put      (Heston) : ', round(error_Puo, 7))
    print('Down and In  put      (Heston) : ', round(option_Pdi, 7),    '  Error Down and In  put      (Heston) : ', round(error_Pdi, 7))
    print('Up   and In  call     (Heston) : ', round(option_Cui, 7),    '  Error Up   and In  call     (Heston) : ', round(error_Cui, 7))
    print('Double barrier option (Heston) : ', round(option_double, 7), '  Error Double barrier option (Heston) : ', round(error_double, 7))

   # # ANALYTICAL
   #
   #
   #  H = lambda_min
   #  K = strike
   #  la = (r + (sigma1 ** 2) / 2) / (sigma1 ** 2)
   #  if H <= K:
   #      y = np.log(H * H / (S0 * K)) / (sigma1 * np.sqrt(T)) + H * sigma1 * np.sqrt(T)
   #      c_di = S0 * np.power(H / S0, 2 * la) * norm.cdf(y) - K * np.exp(-r * T) * np.power(H / S0, 2 * la - 2) * norm.cdf(
   #          y - sigma1 * np.sqrt(T))
   #      c_do = BS_Call - c_di
   #  else:
   #      x1 = np.log(S0 / H) / (sigma1 * np.sqrt(T)) + H * sigma1 * np.sqrt(T)
   #      y1 = np.log(H / S0) / (sigma1 * np.sqrt(T)) + H * sigma1 * np.sqrt(T)
   #      c_do = S0 * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma1 * np.sqrt(T)) - S0 * np.power(H / S0,
   #                                                                                                        2 * la) * norm.cdf(
   #          y1) + K * np.exp(-r * T) * np.power(H / S0, 2 * la - 2) * norm.cdf(y1 - sigma1 * np.sqrt(T))
   #
   #  print('c_do :  ', c_do)


# # PLOTS
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), np.std(Asset_Heston_trj_T, axis=1))
    # plt.xlabel("Time")
    # plt.ylabel('Standar dev')
    # plt.title("Evolution of the standard deviation (Heston)")
    # plt.savefig("Evolution_of_the_standard_deviation(Hest).pdf")
    # plt.show()
    #
    # plt.plot(np.hstack((np.array([0]), maturities)), Asset_Heston_trj_T[:, 0:20])
    # plt.xlabel("Time")
    # plt.ylabel("Asset price")
    # plt.title("Asset trajectories for the whole life of the contract (Hest)")
    # plt.savefig("Asset_trajectories_for_the_whole_life_of_the_contract(Hest).pdf")
    # plt.show()
    #
    # # plt.hist(Asset_Heston_trj_tM[-1], bins=1000, density=True)
    # # plt.title("Density of the asset with Heston at tm = " + str(tM))
    # # plt.savefig("(Hest)Density_of_the_asset_with_Heston_at_tm=" + str(tM) + ".pdf")
    # # plt.show()




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
    print("    %-24s: Number of simulations (2^J J = 22 by default)" % ("-J"))


def run(args):
    # Default values

    J = 22  # Simulation of BS
    seed = 100

    np.random.seed(seed)

    # GBM and Black and Sholes parameters
    sigma = .24
    delta_t = 1 / 52
    # Option data
    T = 1.0  # Option maturity
    tM = 1.0
    tM = np.round(tM, 6)
    T = np.round(T, 6)
    strike = 1.03
    S0 = 1

    lambda_min = 0.7
    lambda_max = 1.15



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

    ################################

    curve = PAR.curve1
    final_TM = ACF.preparingStructure(curve, delta_t, T, tM)

    # COMPUTING THE PRICE OF THE PUT WITH BS
    index_T, index_tM, BS_put, BS_Call, P_0_T = ACF.put_call_price(final_TM, T, tM, S0, strike, sigma)

    P_0_tM = final_TM[index_tM, 1]  # Discount factor P(0,tM)


    # The Martingale property up to the maturity of the option
    # Selecting the time steps from 0 to T. They will be needed for verifing the martingale property
    time_steps_T = final_TM[:index_T + 1,4]

    r = final_TM[index_T, 2]

    if method == 'GBM':
        Barrier_GBM(S0, sigma, time_steps_T, J, strike, P_0_tM, P_0_T, BS_put, tM, lambda_min, lambda_max)
    elif method == 'VG':
        Barrier_VG(S0, time_steps_T, J, strike, P_0_T, BS_put, P_0_tM, lambda_min, lambda_max, BS_Call)
    elif method == 'Heston':
        Barrier_Heston(S0, time_steps_T, J, strike, P_0_T, BS_put, P_0_tM, lambda_min, lambda_max, BS_Call)



if __name__ == "__main__":
    run(sys.argv)

