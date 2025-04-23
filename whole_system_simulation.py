import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 参数设置
high_fs = 5e9         # 采样率 (1 MHz @ 1sample/us)
fs = 1e6
fc = 2.4e9        # 载频 (2.4 GHz)
fr_shift = 250000  # 频率偏移 (250 kHz)
duration = 88e-6   # 信号持续时间 (88 µs)
high_t = np.arange(0, duration, 1/high_fs)  # high resolution时间向量
t = np.arange(0, duration, 1/fs)
sample_indices = (t * high_fs).astype(int)

antenna_interval = 0.06
rx1_position = (-antenna_interval/2, 0)
rx2_position = (antenna_interval/2, 0)
tx1_position = (0, 1)
tx2_position = (-0.8, 1)

initial_phase_tx1 = 0.5
initial_phase_tx2 = 0.2
initial_phase_rx1 = np.pi/3
initial_phase_rx2 = np.pi/6

rx1_drift = 0.000001
rx2_drift = 0.000004

T = 0 #time

frame_sqn = 0

def lowpass_filter(data, cutoff, high_fs, order=1):
    nyquist = 0.5 * high_fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
def up_conversion(base_band_signal_I, base_band_signal_Q, fc, high_t, initial_phase, distance, delta_t):
    print(distance)
    carrier_signal_I = np.cos(2*np.pi*fc*high_t + initial_phase + ((distance%0.125)/0.125*np.pi*2) + (delta_t*fc*np.pi*2))
    carrier_signal_Q = np.sin(2*np.pi*fc*high_t + initial_phase + ((distance%0.125)/0.125*np.pi*2) + (delta_t*fc*np.pi*2))

    #the signal propagate over the channel
    signal = base_band_signal_I*carrier_signal_I - base_band_signal_Q*carrier_signal_Q

    return signal

def down_conversion(signal, fc, high_t, initial_phase, rx_drift, delta_t):
    down_carrier_signal_I = np.cos(2*np.pi*(fc + rx_drift*fc)*high_t + initial_phase + delta_t*(fc + rx_drift*fc)*np.pi*2)
    down_carrier_signal_Q = -np.sin(2*np.pi*(fc + rx_drift*fc)*high_t + initial_phase + delta_t*(fc + rx_drift*fc)*np.pi*2)

    base_band_recover_signal_I = signal*down_carrier_signal_I
    base_band_recover_signal_Q = signal*down_carrier_signal_Q

    base_band_recover_signal_I_lowpass = lowpass_filter(base_band_recover_signal_I, 1e6, high_fs)
    base_band_recover_signal_Q_lowpass = lowpass_filter(base_band_recover_signal_Q, 1e6, high_fs)

    return base_band_recover_signal_I_lowpass, base_band_recover_signal_Q_lowpass


def generate_base_band_signal(high_t, fr_shift, initial_phase):
    base_band_signal_I = np.sin(2*np.pi*fr_shift*high_t + initial_phase)
    base_band_signal_Q = np.cos(2*np.pi*fr_shift*high_t + initial_phase)

    return base_band_signal_I, base_band_signal_Q


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

delta_t = 0.00005
delta_t_error_rate = 0.001
diff_diff_list = []
while frame_sqn < 100:
    print('frame_sqn:', frame_sqn)
    tx1_packet1_I, tx1_packet1_Q = generate_base_band_signal(high_t, fr_shift, initial_phase_tx1)

    tx1_rx1_distance = ((tx1_position[0] - rx1_position[0])**2 + (tx1_position[1] - rx1_position[1])**2)**(0.5)
    tx1_rx2_distance = ((tx1_position[0] - rx2_position[0])**2 + (tx1_position[1] - rx2_position[1])**2)**(0.5)

    tx1_signal_rx1 = up_conversion(tx1_packet1_I, tx1_packet1_Q, fc, high_t, initial_phase_tx1, tx1_rx1_distance, 0)
    tx1_signal_rx2 = up_conversion(tx1_packet1_I, tx1_packet1_Q, fc, high_t, initial_phase_tx1, tx1_rx2_distance, 0)

    rx1_packet1_I, rx1_packet1_Q = down_conversion(tx1_signal_rx1, fc, high_t, initial_phase_rx1, rx1_drift, 0)
    rx2_packet1_I, rx2_packet1_Q = down_conversion(tx1_signal_rx2, fc, high_t, initial_phase_rx2, rx2_drift, 0)


    tx2_packet2_I, tx2_packet2_Q = generate_base_band_signal(high_t+delta_t, fr_shift, initial_phase_tx2)

    tx2_rx1_distance = ((tx2_position[0] - rx1_position[0])**2 + (tx2_position[1] - rx1_position[1])**2)**(0.5)
    tx2_rx2_distance = ((tx2_position[0] - rx2_position[0])**2 + (tx2_position[1] - rx2_position[1])**2)**(0.5)

    tx2_signal_rx1 = up_conversion(tx2_packet2_I, tx2_packet2_Q, fc, high_t, initial_phase_tx2, tx2_rx1_distance, delta_t)
    tx2_signal_rx2 = up_conversion(tx2_packet2_I, tx2_packet2_Q, fc, high_t, initial_phase_tx2, tx2_rx2_distance, delta_t)

    rx1_packet2_I, rx1_packet2_Q = down_conversion(tx2_signal_rx1, fc, high_t, initial_phase_rx1, rx1_drift, delta_t)
    rx2_packet2_I, rx2_packet2_Q = down_conversion(tx2_signal_rx2, fc, high_t, initial_phase_rx2, rx2_drift, delta_t)

    rx1_packet1_I = rx1_packet1_I[sample_indices]
    rx1_packet1_Q = rx1_packet1_Q[sample_indices]
    rx2_packet1_I = rx2_packet1_I[sample_indices]
    rx2_packet1_Q = rx2_packet1_Q[sample_indices]

    rx1_packet2_I = rx1_packet2_I[sample_indices]
    rx1_packet2_Q = rx1_packet2_Q[sample_indices]
    rx2_packet2_I = rx2_packet2_I[sample_indices]
    rx2_packet2_Q = rx2_packet2_Q[sample_indices]

    packet1_phase_diff = cal_packet_phase_diff(rx2_packet1_I, rx2_packet1_Q, rx1_packet1_I, rx1_packet1_Q)
    packet2_phase_diff = cal_packet_phase_diff(rx2_packet2_I, rx2_packet2_Q, rx1_packet2_I, rx1_packet2_Q)

    plt.figure()
    plt.plot(packet2_phase_diff, marker='.')
    plt.show()
    angle = np.arccos((packet2_phase_diff[frame_sqn] + (initial_phase_rx1 - initial_phase_rx2))/(np.pi*2)*0.125/antenna_interval)

    x = 5 * np.cos(angle)
    y = 5 * np.sin(angle)

    plt.figure()
    plt.scatter(rx1_position[0], rx1_position[1], marker='o', color='r', label = 'RX1')
    plt.scatter(rx2_position[0], rx2_position[1], marker='o', color='r', label = 'RX2')
    plt.scatter(tx1_position[0], tx1_position[1], marker='x', color='y', label = 'TX1')
    plt.scatter(tx2_position[0], tx2_position[1], marker='x', color='b', label = 'Target')
    plt.plot([0, x], [0, y])
    plt.xlim(-1.2, 1.2)
    #plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.show()

    diff_diff = packet2_phase_diff - packet1_phase_diff
    for i in range(len(diff_diff)):
        if diff_diff[i] > np.pi:
            diff_diff[i] = diff_diff[i] - 2 * np.pi
        if diff_diff[i] < -np.pi:
            diff_diff[i] = diff_diff[i] + 2 * np.pi
    
    plt.figure()
    plt.plot(diff_diff, marker='.')
    plt.show()

    diff_diff_mean = np.mean(diff_diff[1:])
    diff_diff_list.append(diff_diff_mean)

    phase_change = (2 * np.pi * fc*(1+rx1_drift) * delta_t*(1 + delta_t_error_rate)) - (2 * np.pi * fc*(1+rx2_drift) * delta_t*(1 + delta_t_error_rate)) 
    phase_change_1 = phase_change%(2*np.pi)

    if phase_change_1 > np.pi:
        phase_change_1 = phase_change_1 - 2 * np.pi
    if phase_change_1 < -np.pi:
        phase_change_1 = phase_change_1 + 2 * np.pi
    print(diff_diff_mean, phase_change_1)
    print((diff_diff_mean + phase_change_1)%(2*np.pi))
    phase_diff = (diff_diff_mean + phase_change_1)%(2*np.pi)
    if phase_diff > np.pi:
        phase_diff = phase_diff - 2 * np.pi
    if phase_diff < -np.pi:
        phase_diff = phase_diff + 2 * np.pi
    angle = np.arccos(phase_diff/(np.pi*2)*0.125/antenna_interval)

    x = 5 * np.cos(angle)
    y = 5 * np.sin(angle)

    plt.figure()
    plt.scatter(rx1_position[0], rx1_position[1], marker='o', color='r', label='RX1')
    plt.scatter(rx2_position[0], rx2_position[1], marker='o', color='r', label='RX2')
    plt.scatter(tx1_position[0], tx1_position[1], marker='x', color='y', label='TX1')
    plt.scatter(tx2_position[0], tx2_position[1], marker='x', color='b', label='Target')
    plt.plot([0, x], [0, y])
    plt.xlim(-1.2, 1.2)
    plt.title('Amuna')
    #plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.show()

    frame_sqn = frame_sqn + 1

plt.figure()
plt.hist(diff_diff_list, bins=100)
plt.show()

print(np.mean(diff_diff_list))
#mean_phase_diff = np.mean(diff_diff_list)
mean_phase_diff = packet2_phase_diff[0]
angle = np.arccos((mean_phase_diff + (initial_phase_rx1 - initial_phase_rx2))/(np.pi*2)*0.125/antenna_interval)


x = 5 * np.cos(angle)
y = 5 * np.sin(angle)

plt.figure()
plt.scatter(rx1_position[0], rx1_position[1], marker='o', color='r')
plt.scatter(rx2_position[0], rx2_position[1], marker='o', color='r')
plt.scatter(tx1_position[0], tx1_position[1], marker='x', color='y')
plt.scatter(tx2_position[0], tx2_position[1], marker='x', color='b')
plt.plot([0, x], [0, y])
plt.xlim(-1.2, 1.2)
#plt.ylim(-1.2, 1.2)
plt.show()