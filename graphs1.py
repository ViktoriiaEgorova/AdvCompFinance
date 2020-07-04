import matplotlib.pyplot as plt

J = [10, 12, 14, 16, 18, 20, 22]
# th_call = [0.0820826 for j in J]
# th_put = [0.1135668 for j in J]

th_call = [0.0902802 for j in J]
th_put = [0.1021135 for j in J]

# # VG
# VG_DO = [0.0786328, 0.0849207, 0.0831379, 0.0831164, 0.084006, 0.0840639, 0.0840357]
# VG_DO_error = [0.004204, 0.0021607, 0.00108, 0.0005398, 0.0002728, 0.000136, 6.8e-05]
# VG_UO = [0.1011111, 0.0977074, 0.0958453, 0.097931, 0.0976013, 0.0975987, 0.0976264]
# VG_UO_error = [0.0044788, 0.0022576, 0.0011167, 0.0005607, 0.0002803, 0.0001404, 7.02e-05]
# VG_DI = [0.0594992, 0.0619118, 0.059683, 0.0607728, 0.0606485, 0.0607015, 0.0607244]
# VG_DI_error = [0.0043837, 0.0022566, 0.0011055, 0.0005554, 0.0002776, 0.000139, 6.95e-05]
# VG_UI = [0.0758107, 0.0816171, 0.0795229, 0.079762, 0.080534, 0.0806031, 0.0806039]
# VG_UI_error = [0.0042302, 0.0021784, 0.0010894, 0.0005441, 0.000275, 0.0001371, 6.86e-05]
# VG_double = [0.328772, 0.3294958, 0.3364885, 0.3313501, 0.3325627, 0.3309203, 0.3309554]
# VG_double_error = [0.0138113, 0.0069876, 0.0035211, 0.0017481, 0.000876, 0.0004374, 0.0002187]

# VG
VG_DO = [0.0770638, 0.0832086, 0.0833556, 0.0819601, 0.0823669, 0.0823659, 0.0823499]
VG_DO_error = [0.0041911, 0.0021738, 0.0010952, 0.0005449, 0.0002719, 0.000136, 6.8e-05]
VG_UO = [0.0952566, 0.0950152, 0.0930346, 0.0968517, 0.0958292, 0.0957236, 0.0957823]
VG_UO_error = [0.0045057, 0.0022281, 0.0010984, 0.0005651, 0.0002807, 0.0001402, 7.02e-05]
VG_DI = [0.0587325, 0.0580688, 0.0566021, 0.060121, 0.0594665, 0.0593544, 0.0595345]
VG_DI_error = [0.0044389, 0.0021924, 0.0010815, 0.0005603, 0.0002779, 0.0001388, 6.95e-05]
VG_UI = [0.073893, 0.0795128, 0.0799947, 0.0786961, 0.0790227, 0.0790278, 0.0789738]
VG_UI_error = [0.0042247, 0.0021938, 0.0011044, 0.0005492, 0.0002741, 0.0001371, 6.85e-05]
VG_double = [0.3277772, 0.3229482, 0.3264674, 0.3269964, 0.3259484, 0.3250897, 0.3254043]
VG_double_error = [0.0139912, 0.0069839, 0.0035017, 0.0017504, 0.0008753, 0.0004373, 0.0002188]

# # Heston
# Heston_DO = [0.086296, 0.0855055, 0.0864651, 0.0864002, 0.0854023, 0.085976, 0.0858859]
# Heston_DO_error = [0.0051058, 0.0024435, 0.00124, 0.0006207, 0.0003075, 0.0001544, 7.72e-05]
# Heston_UO = [0.0975114, 0.0998568, 0.100913, 0.101324, 0.1012871, 0.1010713, 0.1010084]
# Heston_UO_error = [0.0042624, 0.0021255, 0.0010687, 0.0005361, 0.0002686, 0.0001342, 6.7e-05]
# Heston_DI = [0.0517151, 0.0509795, 0.0533684, 0.0538314, 0.0541718, 0.0537285, 0.0535585]
# Heston_DI_error = [0.0040206, 0.0019906, 0.0010163, 0.0005108, 0.000256, 0.0001277, 6.37e-05]
# Heston_UI = [0.0840701, 0.0831469, 0.0840342, 0.084157, 0.0831586, 0.0837185, 0.083636]
# Heston_UI_error = [0.0051278, 0.0024553, 0.0012462, 0.0006236, 0.000309, 0.0001551, 7.76e-05]
# Heston_double = [0.3272258, 0.3433774, 0.3416588, 0.3364678, 0.3350164, 0.336005, 0.3367341]
# Heston_double_error = [0.013717, 0.0069246, 0.0034624, 0.0017223, 0.0008609, 0.0004307, 0.0002154]

# Heston
Heston_DO = [0.0855372, 0.083578, 0.0846488, 0.084773, 0.0837478, 0.0843274, 0.0843153]
Heston_DO_error = [0.0051844, 0.0024441, 0.0012365, 0.0006205, 0.0003074, 0.0001544, 7.72e-05 ]
Heston_UO = [0.0949779, 0.0979644, 0.098879, 0.0994291, 0.0993596, 0.0991078, 0.0991057]
Heston_UO_error = [0.0042256, 0.0021242, 0.0010679, 0.0005364, 0.0002685, 0.0001342, 6.7e-05]
Heston_DI = [0.0497168, 0.0500023, 0.052456, 0.0528269, 0.052955, 0.0526969, 0.0526845]
Heston_DI_error = [0.003983, 0.0019918, 0.0010164, 0.000511, 0.0002557, 0.0001276, 6.37e-05]
Heston_UI = [0.0838327, 0.0814305, 0.0822668, 0.0825621, 0.0815243, 0.0821189, 0.0821068]
Heston_UI_error = [0.0052026, 0.0024554, 0.0012427, 0.0006234, 0.0003089, 0.0001551, 7.76e-05]
Heston_double = [0.3177719, 0.335375, 0.3342068, 0.3295804, 0.3298169, 0.3295868, 0.03295638]
Heston_double_error = [0.0136294, 0.0069109, 0.003461, 0.0017219, 0.0008616, 0.0004307, 0.0002154]

########################################################################################################################


# DOWN AND OUT CALL

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.0911],
#        title = 'Prices of Down-and-Out call with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_call, color = 'red', label = 'th vanilla call')
# ax.plot(J, VG_DO, color = 'blue', label = 'Down-and-Out VG call')
# ax.scatter(J, VG_DO, color = 'blue')
# ax.plot(J, Heston_DO, color = 'green', label = 'Down-and-Out Heston call')
# ax.scatter(J, Heston_DO, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# DOWN AND OUT CALL error

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.006],
#        title = 'Error of Down-and-Out call price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_DO_error, color = 'blue', label = 'Down-and-Out VG call error')
# ax.scatter(J, VG_DO_error, color = 'blue')
# ax.plot(J, Heston_DO_error, color = 'green', label = 'Down-and-Out Heston call error')
# ax.scatter(J, Heston_DO_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# UP AND OUT PUT

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.125],
#        title = 'Prices of Up-and-Out put with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_put, color = 'red', label = 'th vanilla put')
# ax.plot(J, VG_UO, color = 'blue', label = 'Up-and-Out VG put')
# ax.scatter(J, VG_UO, color = 'blue')
# ax.plot(J, Heston_UO, color = 'green', label = 'Up-and-Out Heston put')
# ax.scatter(J, Heston_UO, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# UP AND OUT PUT error

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.006],
#        title = 'Error of Up-and-Out put price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_UO_error, color = 'blue', label = 'Up-and-Out VG put error')
# ax.scatter(J, VG_UO_error, color = 'blue')
# ax.plot(J, Heston_UO_error, color = 'green', label = 'Up-and-Out Heston put error')
# ax.scatter(J, Heston_UO_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# DOWN AND IN PUT

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.125],
#        title = 'Prices of Down-and-In put with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_put, color = 'red', label = 'th vanilla put')
# ax.plot(J, VG_DI, color = 'blue', label = 'Down-and-In VG put')
# ax.scatter(J, VG_DI, color = 'blue')
# ax.plot(J, Heston_DI, color = 'green', label = 'Down-and-In Heston put')
# ax.scatter(J, Heston_DI, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# DOWN AND IN PUT error

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.006],
#        title = 'Error of Down-and-In put price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_DI_error, color = 'blue', label = 'Down-and-In VG put error')
# ax.scatter(J, VG_DI_error, color = 'blue')
# ax.plot(J, Heston_DI_error, color = 'green', label = 'Down-and-In Heston put error')
# ax.scatter(J, Heston_DI_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# UP AND IN CALL

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.0911],
#        title = 'Prices of Up-and-In call with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_call, color = 'red', label = 'th vanilla call')
# ax.plot(J, VG_UI, color = 'blue', label = 'Up-and-In VG call')
# ax.scatter(J, VG_UI, color = 'blue')
# ax.plot(J, Heston_UI, color = 'green', label = 'Up-and-In Heston call')
# ax.scatter(J, Heston_UI, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# UP AND IN CALL error

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.006],
#        title = 'Error of Up-and-In call price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_UI_error, color = 'blue', label = 'Up-and-In VG call error')
# ax.scatter(J, VG_UI_error, color = 'blue')
# ax.plot(J, Heston_UI_error, color = 'green', label = 'Up-and-In Heston call error')
# ax.scatter(J, Heston_UI_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# DOUBLE BARRIER

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.5],
#        title = 'Prices of Double Barrier option with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_call, color = 'orange', label = 'th vanilla call')
# ax.plot(J, th_put, color = 'red', label = 'th vanilla put')
# ax.plot(J, VG_double, color = 'blue', label = 'Double Barrier VG option')
# ax.scatter(J, VG_double, color = 'blue')
# ax.plot(J, Heston_double, color = 'green', label = 'Double Barrier Heston option')
# ax.scatter(J, Heston_double, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################

# DOUBLE BARRIER error

# ##################################
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.015],
#        title = 'Error of Double Barrier option price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_double_error, color = 'blue', label = 'Double Barrier VG option error')
# ax.scatter(J, VG_double_error, color = 'blue')
# ax.plot(J, Heston_double_error, color = 'green', label = 'Double Barrier Heston option error')
# ax.scatter(J, Heston_double_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()
#
# ##################################