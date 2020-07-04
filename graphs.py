import matplotlib.pyplot as plt

J = [10, 12, 14, 16, 18, 20, 22]
th_call = [0.0820826 for j in J]
th_put = [0.1135668 for j in J]

VG_call = [0.0513642, 0.0519745, 0.0514311, 0.051198, 0.0514236, 0.0516126, 0.0515242,]
VG_call_error = [0.0023921, 0.0012129, 0.0006077, 0.0003036, 0.0001524, 7.62e-05, 3.81e-05]
Heston_call = [0.0523, 0.05309, 0.05365, 0.05398, 0.05335, 0.0536, 0.05353]
Heston_call_error = [0.00277, 0.00142, 0.00071, 0.00036, 0.00018, 9e-05, 4e-05]

VG_put = [0.06171, 0.0630983, 0.0613545, 0.0627054, 0.0626505, 0.0626337, 0.0626515]
VG_put_error = [0.0027608, 0.001365, 0.0006746, 0.0003413, 0.000171, 8.57e-05, 4.28e-05]
Heston_put = [0.06381, 0.06424, 0.06484, 0.06474, 0.06479, 0.06464, 0.06464]
Heston_put_error = [0.00257, 0.00126, 0.00064, 0.00032, 0.00016, 8e-05, 4e-05]

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.09],
#        title = 'Prices of call with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_call, color = 'red', label = 'th vanilla call')
# ax.plot(J, VG_call, color = 'blue', label = 'Asian VG call')
# ax.scatter(J, VG_call, color = 'blue')
# ax.plot(J, Heston_call, color = 'green', label = 'Asian Heston call')
# ax.scatter(J, Heston_call, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.120],
#        title = 'Prices of put with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Price'
#         )
#
# ax.plot(J, th_put, color = 'red', label = 'th vanilla put')
# ax.plot(J, VG_put, color = 'blue', label = 'Asian VG put')
# ax.scatter(J, VG_put, color = 'blue')
# ax.plot(J, Heston_put, color = 'green', label = 'Asian Heston put')
# ax.scatter(J, Heston_put, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.set(
#        ylim = [0, 0.003],
#        title = 'Error of call price with VG/Heston depending on the trajectory',
#        xlabel = 'Number of trajectories',
#        ylabel = 'Error'
#         )
#
# ax.plot(J, VG_call_error, color = 'blue', label = 'Asian VG call error')
# ax.scatter(J, VG_call_error, color = 'blue')
# ax.plot(J, Heston_call_error, color = 'green', label = 'Asian Heston call error')
# ax.scatter(J, Heston_call_error, color = 'green')
# ax.plot()
# ax.legend()
#
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set(
       ylim = [0, 0.003],
       title = 'Error of put price with VG/Heston depending on the trajectory',
       xlabel = 'Number of trajectories',
       ylabel = 'Error'
        )

ax.plot(J, VG_put_error, color = 'blue', label = 'Asian VG put error')
ax.scatter(J, VG_put_error, color = 'blue')
ax.plot(J, Heston_put_error, color = 'green', label = 'Asian Heston put error')
ax.scatter(J, Heston_put_error, color = 'green')
ax.plot()
ax.legend()

plt.show()

