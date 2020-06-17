import argparse
import numpy as np
import cmath
import random
import math
from my_library import count_parameters
from nn_classes import get_net

def generate_distance(num_users, radius):
    distances = np.zeros(num_users)
    path_loss = np.zeros(num_users)
    pi = np.pi
    for i in range(num_users):
        theta = np.random.rand() * 2 * pi
        r = random.uniform(0, radius ** 2)
        x = math.sin(theta) * (r ** 0.5)
        y = math.cos(theta) * (r ** 0.5)
        distances[i] = (x ** 2 + y ** 2) ** 0.5
        path_loss[i] = 128.1 + 37.6 * math.log10(distances[i] / 1000)
    return path_loss


######################################
######### COMP_TIME.PY ###############
######################################
def rand_comp_time(config,n_clients):
    #Computed empirically on CPU speed 
    # its unit is [seconds/(data_sample*trainable_parameter)]
    MEAN_COMP_RATE = 2.69215782838771e-9
    MEAN_COMP_RATE = 5e-9 
    STD_DEV = 1e-9
    # based on NN and data amount set computation time realization
    comp_rates = np.random.normal(MEAN_COMP_RATE, STD_DEV, size=(n_clients)) # generate a random number with uniform distribution
    comp_rates = np.array([x if x>=0.0 else 0.0 for x in comp_rates])
    n_samples = config["data_per_client"]
    n_parameters = count_parameters(get_net(config))
    n_epochs = config["client_n_epochs"]
    comp_times = comp_rates * n_samples * n_parameters * n_epochs 
    return comp_times

######################################
############# RATE.PY ################
######################################
power_bs = 6.3
power_usr = 0.2
noise_psd = 10 ** (-204/10)
bandwidth = 20 * 1e6
radius = 500

def channel_real(config, n_clients, selected_clients):

    path_loss = generate_distance(n_clients, radius)

    channel = np.zeros(n_clients)
    j = cmath.sqrt(-1)
    for i in range(len(selected_clients)):
        k = selected_clients[i]
        variance = 10 ** (-path_loss[k] / 10)
        channel[k] = (math.sqrt(variance/2) * abs(random.normalvariate(0, 1) + random.normalvariate(0, 1) * j)) ** 2
    return channel

def downlink_latency(args, n_clients, dl_clients, data_size):

    channel = channel_real(args, n_clients, dl_clients)
    worst_channel = min(channel[dl_clients])
    dl_rate = bandwidth * math.log2( 1 + power_bs * worst_channel / (noise_psd * bandwidth))
    dl_latency = data_size / dl_rate
    return dl_latency

# ################# This function generates channel rate for all users ######################
# def channel_instantaneous(args):
#     rate_instantaneous = np.zeros(args.num_users)
#     j = cmath.sqrt(-1)
#     for i in range(args.num_users):
#         variance = 10 ** (-args.path_loss[i] / 10)
#         channel = (math.sqrt(variance/2) * abs(random.normalvariate(0, 1) + random.normalvariate(0, 1) * j)) ** 2
#         rate_instantaneous[i] = args.bandwidth * math.log2(1 + args.power_bs * channel / (args.noise_psd * args.bandwidth))
#     return rate_instantaneous

# ############# This functions computes the average expected rate for all users ########################################
# def channel_avg(args):
#     channel_avg_vec = np.zeros(args.num_users)
#     for ind in range(args.num_users):
#         var_exp = ((10 ** (-args.path_loss[ind] / 10)) / 2) * 2
#         channel_avg_vec[ind] = args.bandwidth*math.log2(1 + args.power_bs * var_exp / (args.noise_psd * args.bandwidth))
#     return channel_avg_vec



"""the variable comptime isn't available as I was writing this code"""
"""I assume it's a matrix of the index and computation time of the users selected for the downlink"""
"""the indices in comptime should be the same as in dl_clients"""
"""assume the first column is the index, and the second column is the computation time"""

def uplink_latency(args, n_clients, dl_clients, comp_time, data_size):
    dl_clients = list(dl_clients)
    comptime=np.zeros((len(dl_clients),2))
    comptime[:,0] = np.asarray(dl_clients) # first column client indicies
    comptime[:,1] = comp_time # second column computation times
    num_users_ul = len(dl_clients)
    
    ul_clients = np.zeros( num_users_ul ) + 10000# to save the indices of clients selected in uplink

    # generate the channel
    channel = channel_real(args, n_clients, dl_clients)

    # the array of user indices and corresponding rates in TDMA
    rates = [[0 for i in range(2)] for j in range(len(dl_clients))]
    for i in range(len(dl_clients)):
        rates[i][0] = dl_clients[i] # first column stores the indices
        k = dl_clients[i]
        # rates[i][1] = (i+1) * 1e5
        # rates[0][1] = 1e3
        # rates[1][1] = 1e5
        rates[i][1] = bandwidth * math.log2( 1 + power_usr * channel[k] / (noise_psd * bandwidth) ) # second column stores rates

    # the array of user indices and remaining bits to transmit
    bits_left = [[0 for i in range(2)] for j in range(len(dl_clients))]
    time_left = [[0 for i in range(2)] for j in range(len(dl_clients))]
    for i in range(len(dl_clients)):
        bits_left[i][0] = dl_clients[i]
        bits_left[i][1] = data_size
        time_left[i][0] = dl_clients[i]
        time_left[i][1] = data_size/rates[i][1]


    # only comptime and temp_time_left are sorted in the following
    comptime = sorted(comptime, key=lambda x:x[1]) # sort according to column 2, which is the computation time
    latency = 0
    index = 0 # to count how many users have finished communication
    for i in range(len(dl_clients)): # go through all the users until we can select enough users or until we have gone through all of them

        # k = comptime[i][0] # index of the (i+1)-th user who finishes computation
        if i==len(dl_clients)-1:
            time_available = 1e20 # to make it large enough. meaning: if all the users have finished computation, the transmission will not be paused, and will finish, then we will choose another user to transmit
        else:
            time_available = comptime[i+1][1] - comptime[i][1] # time for THE selected (ONE) user to communicate before a new user is available to communicate

        # find the best user who requires the minimal communication time
        cal_time_left = [[0 for p in range(2)] for q in range(i+1)] # only consider those who have finished computation
        for m in range(i+1): # among all the users who have finished computation
            user_selected = comptime[m][0] # user index in the short vector is n
            n = dl_clients.index(user_selected)
            cal_time_left[m][0] = user_selected
            cal_time_left[m][1] = bits_left[n][1] / rates[n][1]
        cal_time_left = sorted(cal_time_left, key=lambda x:x[1])
        for k in range( i+1 ): # only choose from those who have finished computation
            if cal_time_left[k][1]>0: # exclude those who have finished communication
                user_selected = cal_time_left[k][0] # here we know which user has the mininum remaining communication time among those who have finished computation
                break

        m = dl_clients.index(user_selected) # find the corresponding index
        time_transmission = bits_left[m][1] / rates[m][1]

        bits_left[m][1] = bits_left[m][1] - time_available * rates[m][1]

        time_left[m][1] = time_left[m][1] - time_available

        # if this user doesn't have remaining bits to transmit, consider this user as the one who has completed communication
        if bits_left[m][1] <= 0:
            bits_left[m][1]=0
            time_left[m][1]=0
            ul_clients[index] = user_selected
            index = index+1
        if index == num_users_ul:
            latency = comptime[i][1] + time_transmission
            break # here we have got all the users in the uplink

    # here all the users have finished computation
    if index < num_users_ul:
        latency = comptime[len(dl_clients)-1][1] + time_transmission # if not enough users are selected in uplink even after all the users have finished local computation
    while index < num_users_ul: # here we have all the users available to transmit, except those who have finished communication
        for m in range(len(dl_clients)):  # among all the users who have finished computation
            user_selected = comptime[m][0]  # user index in the short vector is n
            n = dl_clients.index(user_selected) # here user 1 has finished transmission
            cal_time_left[m][0] = user_selected
            cal_time_left[m][1] = bits_left[n][1] / rates[n][1] # calculate all the users' remaining comm time
        cal_time_left = sorted(cal_time_left, key=(lambda x: x[1]))
        for k in range(len(dl_clients)):
            if cal_time_left[k][1] > 0:  # exclude those who have finished communication
                user_selected = cal_time_left[k][0]

                m = dl_clients.index(user_selected)  # find the corresponding index
                ul_clients[index] = user_selected
                time_transmission = bits_left[m][1] / rates[m][1]
                latency = latency + time_transmission
                bits_left[m][1] = 0
                time_left[m][1] = 0
                index = index + 1
                break

                # break

            # m = dl_clients.index(user_selected)  # find the corresponding index
            # ul_clients[index] = user_selected
            # time_transmission = bits_left[m][1] / rates[m][1]
            # latency = latency + time_transmission
            # bits_left[m][1] = 0
            # time_left[m][1] = 0
            # index = index + 1

    return latency

        # ul_clients[i] = comptime.index(min(comptime))
        # comptime[:] = [x-min(comptime) for x in comptime] # calculate the remaining computation time of all the users
