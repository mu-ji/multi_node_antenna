import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

data1 = np.load('100_data_10_degree1.npz')
data2 = np.load('100_data_10_degree2.npz')

#data1 = np.load('1000_data_1.npz')
#data2 = np.load('1000_data_2.npz')

rx1_packet1_I = data1['packet_1_I_data']
rx1_packet1_Q = data1['packet_1_Q_data']

rx1_packet2_I = data1['packet_2_I_data']
rx1_packet2_Q = data1['packet_2_Q_data']

#interval1 = data1['interval']

rx2_packet1_I = data2['packet_1_I_data']
rx2_packet1_Q = data2['packet_1_Q_data']

rx2_packet2_I = data2['packet_2_I_data']
rx2_packet2_Q = data2['packet_2_Q_data']

#interval2 = data2['interval']

def norm(I_data, Q_data):
    amplitude = np.sqrt(I_data ** 2 + Q_data ** 2)
    if amplitude != 0:
        norm_I = I_data / amplitude
        norm_Q = Q_data / amplitude
    else:
        norm_I = 0
        norm_Q = 0
    return norm_I, norm_Q

def calculate_angle(I1, Q1, I2, Q2):
    dot_product = I1 * I2 + Q1 * Q2
    if dot_product>1:
        dot_product = 1
    elif dot_product<-1:
        dot_product = -1

    theta = np.arccos(dot_product)
    cross_product = I1 * Q2 - Q1 * I2
    if cross_product > 0:
        return theta
    else:
        return -theta

def normalization(I, Q):
    for i in range(len(I)):
        norm_I, norm_Q = norm(I[i], Q[i])
        I[i] = norm_I
        Q[i] = norm_Q
    return I, Q
def cal_packet_phase_diff(rx1_packet_I, rx1_packet_Q, rx2_packet_I, rx2_packet_Q):
    rx1_packet1_I, rx1_packet1_Q = normalization(rx1_packet_I, rx1_packet_Q)
    rx2_packet1_I, rx2_packet1_Q = normalization(rx2_packet_I, rx2_packet_Q)
    phase_diff = np.zeros(len(rx1_packet1_I))

    for i in range(len(rx1_packet1_I)):
        phase_diff[i] = calculate_angle(rx1_packet1_I[i], rx1_packet1_Q[i], rx2_packet1_I[i], rx2_packet1_Q[i])

    return phase_diff

diff_diff_list = []
for i in range(100):
    packet1_phase_diff = cal_packet_phase_diff(rx1_packet1_I[i], rx1_packet1_Q[i], rx2_packet1_I[i], rx2_packet1_Q[i])
    packet2_phase_diff = cal_packet_phase_diff(rx1_packet2_I[i], rx1_packet2_Q[i], rx2_packet2_I[i], rx2_packet2_Q[i])

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff')
    # axs[0].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff')
    # axs[0].set_xlabel('time (us)')
    # axs[0].set_ylabel('phase')
    # axs[0].set_title('packet phase diff')
    # axs[0].legend()

    # axs[1].plot(packet2_phase_diff - packet1_phase_diff, marker='.', label = '1st packet phase diff - 2nd packet phase diff')
    # plt.show()

    diff_diff = packet2_phase_diff - packet1_phase_diff
    for i in range(len(diff_diff)):
        if diff_diff[i] > np.pi:
            diff_diff[i] = diff_diff[i] - 2 * np.pi
        if diff_diff[i] < -np.pi:
            diff_diff[i] = diff_diff[i] + 2 * np.pi
    
    diff_diff_mean = np.mean(diff_diff)
    diff_diff_list.append(diff_diff_mean)

plt.figure()
plt.hist(diff_diff_list, bins=100)
plt.show()
