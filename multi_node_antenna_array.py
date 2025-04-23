import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

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
axs[0].plot(np.arctan2(rx1_packet1_Q[0], rx1_packet1_I[0]) - np.arctan2(rx2_packet1_Q[0], rx2_packet1_I[0]), marker='.', label = '1st packet phase diff')
axs[0].plot(np.arctan2(rx1_packet2_Q[0], rx1_packet2_I[0]) - np.arctan2(rx2_packet2_Q[0], rx2_packet2_I[0]), marker='.', label = '2nd packet phase diff')
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel('phase')
axs[0].set_title('first attampt')
axs[0].legend()

axs[1].plot(np.arctan2(rx1_packet1_Q[1], rx1_packet1_I[1]) - np.arctan2(rx2_packet1_Q[1], rx2_packet1_I[1]), marker='.', label = '1st packet phase diff')
axs[1].plot(np.arctan2(rx1_packet2_Q[1], rx1_packet2_I[1]) - np.arctan2(rx2_packet2_Q[1], rx2_packet2_I[1]), marker='.', label = '2nd packet phase diff')
axs[1].set_xlabel('time (us)')
axs[1].set_ylabel('phase')
axs[1].set_title('second attampt')
axs[1].legend()

plt.tight_layout()
plt.show() 


data1 = np.load('1000_data_1.npz')
data2 = np.load('1000_data_2.npz')

data1 = np.load('100_data_10_degree1.npz')
data2 = np.load('100_data_10_degree2.npz')

rx1_packet1_I = data1['packet_1_I_data']
rx1_packet1_Q = data1['packet_1_Q_data']

rx1_packet2_I = data1['packet_2_I_data']
rx1_packet2_Q = data1['packet_2_Q_data']

interval1 = data1['interval']

rx2_packet1_I = data2['packet_1_I_data']
rx2_packet1_Q = data2['packet_1_Q_data']

rx2_packet2_I = data2['packet_2_I_data']
rx2_packet2_Q = data2['packet_2_Q_data']

interval2 = data2['interval']


interval = (interval1+interval2)/2
def compensate_phase_diff(phase_diff): 
    for i in range(len(phase_diff)-1):
        while (phase_diff[i+1] - phase_diff[i] > np.pi):
        #if phase_diff[i+1] - phase_diff[i] > np.pi:
            phase_diff[i+1] = phase_diff[i+1] - 2*np.pi
            
        while (phase_diff[i+1] - phase_diff[i] < -np.pi):
        #if phase_diff[i+1] - phase_diff[i] < -np.pi:
            phase_diff[i+1] = phase_diff[i+1] + 2*np.pi

    # for i in range(len(phase_diff)):
    #     if phase_diff[i] > np.pi:
    #         phase_diff[i] = phase_diff[i] - 2*np.pi
    #     if phase_diff[i] < -np.pi:
    #         phase_diff[i] = phase_diff[i] + 2*np.pi
    
    # for i in range(len(phase_diff)-1):
    #     if phase_diff[i+1] - phase_diff[i] > np.pi:
    #         phase_diff[i+1] = phase_diff[i+1] - 2*np.pi
    #     if phase_diff[i+1] - phase_diff[i] < -np.pi:
    #         phase_diff[i+1] = phase_diff[i+1] + 2*np.pi
    #
    return phase_diff

def cal_slope(phase_diff):
    x = np.arange(len(phase_diff)).reshape(-1, 1)
    y = phase_diff
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]

diff_list = []
for i in range(100):

    packet1_phase_diff = np.arctan2(rx1_packet1_Q[i], rx1_packet1_I[i]) - np.arctan2(rx2_packet1_Q[i], rx2_packet1_I[i])
    packet2_phase_diff = np.arctan2(rx1_packet2_Q[i], rx1_packet2_I[i]) - np.arctan2(rx2_packet2_Q[i], rx2_packet2_I[i])

    packet1_phase_diff_copy = packet1_phase_diff.copy()
    packet2_phase_diff_copy = packet2_phase_diff.copy()
    # fig, axs = plt.subplots(2, 1)

    # axs[0].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff')
    # axs[0].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff')
    # axs[0].set_xlabel('time (us)')
    # axs[0].set_ylabel('phase')
    # axs[0].set_title('packet phase diff')
    # axs[0].legend()

    packet1_phase_diff_compensate = compensate_phase_diff(packet1_phase_diff)
    packet2_phase_diff_compensate = compensate_phase_diff(packet2_phase_diff)

    slope1 = cal_slope(packet1_phase_diff_compensate)
    slope2 = cal_slope(packet2_phase_diff_compensate)
    slope = (slope1 + slope2)/2
    # axs[1].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff after compensate')
    # axs[1].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff after compensate')
    # axs[1].set_xlabel('time (us)')
    # axs[1].set_ylabel('phase')
    # axs[1].set_title('packet phase diff after compensate')
    # axs[1].legend()
    # plt.tight_layout()
    # plt.show()

    # if np.mean(packet2_phase_diff_compensate - packet1_phase_diff_compensate) < -2*np.pi:

    #     fig, axs = plt.subplots(2, 1)

    #     axs[0].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff')
    #     axs[0].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff')
    #     axs[0].set_xlabel('time (us)')
    #     axs[0].set_ylabel('phase')
    #     axs[0].set_title('packet phase diff')
    #     axs[0].legend()
    #     axs[1].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff after compensate')
    #     axs[1].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff after compensate')
    #     axs[1].set_xlabel('time (us)')
    #     axs[1].set_ylabel('phase')
    #     axs[1].set_title('packet phase diff after compensate')
    #     axs[1].legend()
    #     plt.tight_layout()
    #     plt.show()
    #print(interval[i])
    #print((interval[i]/16000000*1000000)*slope)
    phase_change = interval[i]/16*slope%(2*np.pi)
    diff_diff = (packet2_phase_diff_compensate) - packet1_phase_diff_compensate
    #diff_diff = (packet2_phase_diff_compensate) - packet1_phase_diff_compensate

    diff_diff_mean = np.mean(diff_diff)
    if diff_diff_mean > np.pi:
        diff_diff_mean = diff_diff_mean - 2*np.pi
    elif diff_diff_mean < -np.pi:
        diff_diff_mean = diff_diff_mean + 2*np.pi

    phase_diff = (diff_diff_mean + phase_change)%(2*np.pi)
    if phase_diff > np.pi:
        phase_diff = phase_diff - 2 * np.pi
    if phase_diff < -np.pi:
        phase_diff = phase_diff + 2 * np.pi

    # if diff_diff_mean < 0:
    #     fig, axs = plt.subplots(2, 1)

    #     axs[0].plot(packet1_phase_diff_copy, marker='.', label = '1st packet phase diff')
    #     axs[0].plot(packet2_phase_diff_copy, marker='.', label = '2nd packet phase diff')
    #     axs[0].set_xlabel('time (us)')
    #     axs[0].set_ylabel('phase')
    #     axs[0].set_title('packet phase diff')
    #     axs[0].set_ylim(-7,7)
    #     axs[0].legend()

    #     axs[1].plot(packet1_phase_diff, marker='.', label = '1st packet phase diff after compensate')
    #     axs[1].plot(packet2_phase_diff, marker='.', label = '2nd packet phase diff after compensate')
    #     axs[1].set_xlabel('time (us)')
    #     axs[1].set_ylabel('phase')
    #     axs[1].set_title('packet phase diff after compensate')
    #     axs[1].set_ylim(-7,7)
    #     axs[1].legend()
    #     plt.tight_layout()
    #     plt.show()


    # plt.plot(diff_diff)
    # plt.show()

    diff_list.append(phase_diff)
        
# for i in range(len(diff_list)):
#     if diff_list[i] > np.pi:
#         diff_list[i] = diff_list[i] - 2*np.pi

plt.hist(diff_list,100)
plt.ylabel('count')
plt.xlabel('two packet phase diff')
plt.show()

plt.plot(diff_list, marker='.')
plt.ylabel('count')
plt.xlabel('two packet phase diff') 
plt.show()

print(diff_list)

from sklearn.mixture import GaussianMixture

# 使用 GMM 拟合数据
gmm = GaussianMixture(n_components=2)
gmm.fit(np.array(diff_list).reshape(-1,1))

# 获取均值和方差
means = gmm.means_
covariances = gmm.covariances_

print(f"第一高斯分布: 均值 = {means[0][0]}, 方差 = {covariances[0][0][0]}")
print(f"第二高斯分布: 均值 = {means[1][0]}, 方差 = {covariances[1][0][0]}")

x1 = means[0][0]/6.28*12.5/6
x2 = means[1][0]/6.28*12.5/6
print('x1:', x1, 'x2:', x2)
print('angle1:', np.arccos(x1)/np.pi*180)
print('angle2:', np.arccos(x2)/np.pi*180)