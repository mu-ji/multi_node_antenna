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

base_band_signal_I = np.cos(2*np.pi*fr_shift*high_t)
base_band_signal_Q = np.sin(2*np.pi*fr_shift*high_t)

#上变频
carrier_signal_I = np.cos(2*np.pi*fc*high_t)
carrier_signal_Q = np.sin(2*np.pi*fc*high_t)

#the signal propagate over the channel
signal = base_band_signal_I*carrier_signal_I - base_band_signal_Q*carrier_signal_Q

#下变频
down_carrier_signal_I = np.cos(2*np.pi*fc*high_t)
down_carrier_signal_Q = -np.sin(2*np.pi*fc*high_t)

base_band_recover_signal_I = signal*down_carrier_signal_I
base_band_recover_signal_Q = signal*down_carrier_signal_Q

def lowpass_filter(data, cutoff, high_fs, order=1):
    nyquist = 0.5 * high_fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

base_band_recover_signal_I_lowpass = lowpass_filter(base_band_recover_signal_I, 1e6, high_fs)
base_band_recover_signal_Q_lowpass = lowpass_filter(base_band_recover_signal_Q, 1e6, high_fs)

plt.figure()
plt.plot(high_t, base_band_recover_signal_I_lowpass, marker='.')
plt.plot(high_t, base_band_recover_signal_Q_lowpass, marker='.')
plt.show()

sample_indices = (t * high_fs).astype(int)

fig, axs = plt.subplots(3, 1)
axs[0].plot(t, base_band_signal_I[sample_indices], marker='.')
axs[0].plot(t, base_band_signal_Q[sample_indices], marker='.')
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel('amplitude')
axs[0].set_title('base band signal')

axs[1].plot(t, base_band_recover_signal_I_lowpass[sample_indices], marker='.')
axs[1].plot(t, base_band_recover_signal_Q_lowpass[sample_indices], marker='.')
axs[1].set_xlabel('time (us)')
axs[1].set_ylabel('amplitude')
axs[1].set_title('received base band signal')

axs[2].plot(t, np.arctan2(base_band_signal_Q[sample_indices], base_band_signal_I[sample_indices]), marker='.', label='base band signal')
axs[2].plot(t, np.arctan2(base_band_recover_signal_Q_lowpass[sample_indices], base_band_recover_signal_I_lowpass[sample_indices]), marker='.', label='received base band signal')
axs[2].set_xlabel('time (us)')
axs[2].set_ylabel('phase')
axs[2].set_title('transmitted and received base band signal phase')
plt.tight_layout()
plt.show()

def up_conversion(base_band_signal_I, base_band_signal_Q, fc, high_t, initial_phase):
    carrier_signal_I = np.cos(2*np.pi*fc*high_t + initial_phase)
    carrier_signal_Q = np.sin(2*np.pi*fc*high_t + initial_phase)

    #the signal propagate over the channel
    signal = base_band_signal_I*carrier_signal_I - base_band_signal_Q*carrier_signal_Q

    return signal

def down_conversion(signal, fc, high_t, initial_phase, rx_drift):
    down_carrier_signal_I = np.cos(2*np.pi*(fc + rx_drift*fc)*high_t + initial_phase)
    down_carrier_signal_Q = -np.sin(2*np.pi*(fc + rx_drift*fc)*high_t + initial_phase)

    base_band_recover_signal_I = signal*down_carrier_signal_I
    base_band_recover_signal_Q = signal*down_carrier_signal_Q

    base_band_recover_signal_I_lowpass = lowpass_filter(base_band_recover_signal_I, 1e6, high_fs)
    base_band_recover_signal_Q_lowpass = lowpass_filter(base_band_recover_signal_Q, 1e6, high_fs)

    return base_band_recover_signal_I_lowpass, base_band_recover_signal_Q_lowpass


def generate_base_band_signal(high_t, fr_shift, initial_phase):
    base_band_signal_I = np.sin(2*np.pi*fr_shift*high_t + initial_phase)
    base_band_signal_Q = np.cos(2*np.pi*fr_shift*high_t + initial_phase)

    return base_band_signal_I, base_band_signal_Q

initial_phase_tx1_base = 0.1
initial_phase_tx1 = 0
initial_phase_rx1 = 0
initial_phase_rx2 = np.pi/3

rx1_dirft = 0.000001
rx2_dirft = 0.000004

base_band_signal_tx1_I, base_band_signal_tx1_Q = generate_base_band_signal(high_t, fr_shift, initial_phase_tx1_base)

tx1_signal = up_conversion(base_band_signal_tx1_I, base_band_signal_tx1_Q, fc, high_t, initial_phase_tx1)

r1_signal_I, r1_signal_Q = down_conversion(tx1_signal, fc, high_t, initial_phase_rx1, rx1_dirft)

base_band_recover_signal_I_lowpass_fft = np.fft.fft(r1_signal_I)
base_band_recover_signal_I_lowpass_fft_freq = np.fft.fftfreq(len(base_band_recover_signal_I_lowpass), d=1/high_fs)
plt.figure()
plt.plot(base_band_recover_signal_I_lowpass_fft_freq, np.abs(base_band_recover_signal_I_lowpass_fft), marker='.')
plt.show()

r2_signal_I, r2_signal_Q = down_conversion(tx1_signal, fc, high_t, initial_phase_rx2, rx2_dirft)

r1_phase = np.arctan2(r1_signal_Q, r1_signal_I)
r2_phase = np.arctan2(r2_signal_Q, r2_signal_I)


fig, axs = plt.subplots(2, 1)
axs[0].plot(t, r1_phase[sample_indices], marker='.')
axs[0].plot(t, r2_phase[sample_indices], marker='.')

axs[1].plot(t, r1_phase[sample_indices] - r2_phase[sample_indices], marker='.')
axs[1].set_ylim(-2*np.pi, 2*np.pi)
plt.show()


