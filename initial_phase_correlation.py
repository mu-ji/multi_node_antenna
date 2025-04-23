import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('100_data_1.npz')
data2 = np.load('100_data_2.npz')

data1 = np.load('100_data_10_degree1.npz')
data2 = np.load('100_data_10_degree2.npz')

rx1_packet1_I = data1['packet_1_I_data']
rx1_packet1_Q = data1['packet_1_Q_data']

rx1_packet2_I = data1['packet_2_I_data']
rx1_packet2_Q = data1['packet_2_Q_data']

rx2_packet1_I = data2['packet_1_I_data']
rx2_packet1_Q = data2['packet_1_Q_data']

rx2_packet2_I = data2['packet_2_I_data']
rx2_packet2_Q = data2['packet_2_Q_data']

print(rx1_packet1_I.shape)
initial_phase_list = []
end_phase_list = []

for i in range(99):
    diff = np.arctan2(rx1_packet1_Q[i][0], rx1_packet1_I[i][0]) - np.arctan2(rx2_packet1_Q[i][0], rx2_packet1_I[i][0])
    if diff > 2*np.pi:
        diff = diff - 2*np.pi
    if diff < -2*np.pi:
        diff = diff + 2*np.pi

    initial_phase_list.append(diff)


for i in range(1, 100):
    diff = np.arctan2(rx1_packet2_Q[i][87], rx1_packet2_I[i][87]) - np.arctan2(rx2_packet2_Q[i][87], rx2_packet2_I[i][87])
    if diff > 2*np.pi:
        diff = diff - 2*np.pi
    if diff < -2*np.pi:
        diff = diff + 2*np.pi
    end_phase_list.append(diff)

plt.plot(initial_phase_list, end_phase_list, marker='.', label = '1st packet phase diff')
plt.show()

print(np.array(initial_phase_list).shape)
correlation_coefficient = np.corrcoef(initial_phase_list, end_phase_list)[0, 1]
print(correlation_coefficient)

