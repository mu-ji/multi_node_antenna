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

phase = np.pi/2
base_band_sigal_I = np.cos(2*np.pi*fr_shift*high_t + phase)
base_band_sigal_Q = np.sin(2*np.pi*fr_shift*high_t + phase)

fig, axs = plt.subplots(2, 1)
axs[0].plot(high_t, base_band_sigal_I, marker='.')
axs[0].plot(high_t, base_band_sigal_Q, marker='.')
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel('amplitude')
axs[0].set_title('base band signal IQ')

axs[1].plot(high_t, np.arctan2(base_band_sigal_Q, base_band_sigal_I), marker='.')
axs[1].set_xlabel('time (us)')
axs[1].set_ylabel('phase')
axs[1].set_title('base band signal phase')

plt.show()

#上变频

carrier_sigal_I = np.cos(2*np.pi*fc*high_t)
carrier_sigal_Q = np.sin(2*np.pi*fc*high_t)

signal = base_band_sigal_I*carrier_sigal_I - base_band_sigal_Q*carrier_sigal_Q

signal_fft = np.fft.fft(signal)
signal_fft_freq = np.fft.fftfreq(len(signal), d=1/high_fs)

fig, axs = plt.subplots(2, 1)
axs[0].plot(high_t, signal, marker='.')
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel('amplitude')

axs[1].plot(signal_fft_freq, np.abs(signal_fft), marker='.')
axs[1].set_xlabel('frequency (Hz)')
axs[1].set_ylabel('amplitude')
plt.show()

#下变频

down_carrier_sigal_I = np.cos(2*np.pi*fc*high_t)
down_carrier_sigal_Q = -np.sin(2*np.pi*fc*high_t)

base_band_recover_signal_I = signal*down_carrier_sigal_I
base_band_recover_signal_Q = signal*down_carrier_sigal_Q

fft = np.fft.fft(base_band_recover_signal_Q)
fft_freq = np.fft.fftfreq(len(base_band_recover_signal_Q), d=1/high_fs)

fig, axs = plt.subplots(2, 1)
axs[0].plot(high_t, base_band_recover_signal_Q, marker='.')
axs[0].set_xlabel('time (us)')
axs[0].set_ylabel('amplitude')

axs[1].plot(fft_freq, np.abs(fft), marker='.')
axs[1].set_xlabel('frequency (Hz)')
axs[1].set_ylabel('amplitude')
plt.show()

def lowpass_filter(data, cutoff, high_fs, order=5):
    nyquist = 0.5 * high_fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

base_band_recover_signal_I_lowpass = lowpass_filter(base_band_recover_signal_I, 1e6, high_fs)
base_band_recover_signal_I_lowpass_fft = np.fft.fft(base_band_recover_signal_I_lowpass)
base_band_recover_signal_I_lowpass_fft_freq = np.fft.fftfreq(len(base_band_recover_signal_I_lowpass), d=1/high_fs)

base_band_recover_signal_Q_lowpass = lowpass_filter(base_band_recover_signal_Q, 1e6, high_fs)
base_band_recover_signal_Q_lowpass_fft = np.fft.fft(base_band_recover_signal_Q_lowpass)
base_band_recover_signal_Q_lowpass_fft_freq = np.fft.fftfreq(len(base_band_recover_signal_Q_lowpass), d=1/high_fs)

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(high_t, base_band_recover_signal_I_lowpass, marker='.')
axs[0,0].set_xlabel('time (us)')
axs[0,0].set_ylabel('amplitude')

axs[0,1].plot(high_t, base_band_recover_signal_Q_lowpass, marker='.')
axs[0,1].set_xlabel('time (us)')
axs[0,1].set_ylabel('amplitude')

axs[1,0].plot(base_band_recover_signal_I_lowpass_fft_freq, np.abs(base_band_recover_signal_I_lowpass_fft), marker='.')
axs[1,0].set_xlabel('frequency (Hz)')
axs[1,0].set_ylabel('amplitude')

axs[1,1].plot(base_band_recover_signal_Q_lowpass_fft_freq, np.abs(base_band_recover_signal_Q_lowpass_fft), marker='.')
axs[1,1].set_xlabel('frequency (Hz)')
axs[1,1].set_ylabel('amplitude')
plt.show()

