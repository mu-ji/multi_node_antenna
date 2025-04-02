import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('2_data_1.npz')
data2 = np.load('2_data_2.npz')

rx1_packet1_I = data1['packet_1_I_data']
rx1_packet1_Q = data1['packet_1_Q_data']

rx1_packet2_I = data1['packet_2_I_data']
rx1_packet2_Q = data1['packet_2_Q_data']

rx2_packet1_I = data2['packet_1_I_data']
rx2_packet1_Q = data2['packet_1_Q_data']

rx2_packet2_I = data2['packet_2_I_data']
rx2_packet2_Q = data2['packet_2_Q_data']

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.arctan2(rx1_packet1_Q[0], rx1_packet1_I[0]) - np.arctan2(rx2_packet1_Q[0], rx2_packet1_I[0]), marker='.')
axs[0].plot(np.arctan2(rx1_packet2_Q[0], rx1_packet2_I[0]) - np.arctan2(rx2_packet2_Q[0], rx2_packet2_I[0]), marker='.')

axs[1].plot(np.arctan2(rx1_packet1_Q[1], rx1_packet1_I[1]) - np.arctan2(rx2_packet1_Q[1], rx2_packet1_I[1]), marker='.')
axs[1].plot(np.arctan2(rx1_packet2_Q[1], rx1_packet2_I[1]) - np.arctan2(rx2_packet2_Q[1], rx2_packet2_I[1]), marker='.')
plt.show()


data1 = np.load('100_data_1.npz')
data2 = np.load('100_data_2.npz')

rx1_packet1_I = data1['packet_1_I_data']
rx1_packet1_Q = data1['packet_1_Q_data']

rx1_packet2_I = data1['packet_2_I_data']
rx1_packet2_Q = data1['packet_2_Q_data']

rx2_packet1_I = data2['packet_1_I_data']
rx2_packet1_Q = data2['packet_1_Q_data']

rx2_packet2_I = data2['packet_2_I_data']
rx2_packet2_Q = data2['packet_2_Q_data']


def compensate_phase_diff(phase_diff): 

    for i in range(len(phase_diff)):
        if phase_diff[i] > np.pi:
            phase_diff[i] = phase_diff[i] - 2*np.pi
        if phase_diff[i] < -np.pi:
            phase_diff[i] = phase_diff[i] + 2*np.pi
    
    for i in range(len(phase_diff)-1):
        if phase_diff[i+1] - phase_diff[i] > np.pi:
            phase_diff[i+1] = phase_diff[i+1] - 2*np.pi
        if phase_diff[i+1] - phase_diff[i] < -np.pi:
            phase_diff[i+1] = phase_diff[i+1] + 2*np.pi

    return phase_diff

diff_list = []
for i in range(100):
    packet1_phase_diff = np.arctan2(rx1_packet1_Q[i], rx1_packet1_I[i]) - np.arctan2(rx2_packet1_Q[i], rx2_packet1_I[i])
    packet2_phase_diff = np.arctan2(rx1_packet2_Q[i], rx1_packet2_I[i]) - np.arctan2(rx2_packet2_Q[i], rx2_packet2_I[i])

    plt.plot(packet1_phase_diff, marker='.')
    plt.plot(packet2_phase_diff, marker='.')
    plt.show()

    packet1_phase_diff = compensate_phase_diff(packet1_phase_diff)
    packet2_phase_diff = compensate_phase_diff(packet2_phase_diff)

    # plt.plot(packet1_phase_diff)
    # plt.plot(packet2_phase_diff)
    # plt.show()

    diff_diff = packet2_phase_diff - packet1_phase_diff

    # plt.plot(diff_diff)
    # plt.show()


    # plt.plot(diff_diff)
    # plt.show()

    diff_list.append(np.mean(diff_diff))
        
# for i in range(len(diff_list)):
#     if diff_list[i] > np.pi:
#         diff_list[i] = diff_list[i] - 2*np.pi

plt.hist(diff_list,100)
plt.show()

print(diff_list)

