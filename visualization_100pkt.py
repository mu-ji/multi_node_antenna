import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('100_data_1.npz')
data2 = np.load('100_data_2.npz')

index = 2

I_data1 = data1['I_data'][index]
Q_data1 = data1['Q_data'][index]

I_data2 = data2['I_data'][index]
Q_data2 = data2['Q_data'][index]

plt.figure()
plt.plot(np.arctan2(Q_data1, I_data1) - np.arctan2(Q_data2, I_data2), marker='.')
plt.show()

plt.figure()
plt.plot(np.arctan2(Q_data1, I_data1) + np.arctan2(Q_data2, I_data2), marker='.')
plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(np.arctan2(Q_data1, I_data1))
axs[0].plot(np.arctan2(Q_data2, I_data2))

axs[1].plot(np.arctan2(Q_data1, I_data1) - np.arctan2(Q_data2, I_data2), marker='.')
axs[1].set_title('phase difference')

axs[2].plot(np.arctan2(Q_data1, I_data1) + np.arctan2(Q_data2, I_data2), marker='.')
axs[2].set_title('phase sumb')

plt.tight_layout()
plt.show()

phase0 = np.arctan2(Q_data1[0], I_data1[0]) - np.arctan2(Q_data2[0], I_data2[0])
phase80 = np.arctan2(Q_data1[80], I_data1[80]) - np.arctan2(Q_data2[80], I_data2[80])
print(phase0)
print(phase80)
if phase80 < phase0:
    phase80 = phase80 + 2*np.pi

print(phase80 - phase0)

phase_diff_list = []
phase_0_list = []
for i in range(len(data1['I_data'])):

    I_data1 = data1['I_data'][i]
    Q_data1 = data1['Q_data'][i]

    I_data2 = data2['I_data'][i]
    Q_data2 = data2['Q_data'][i]

    phase0 = np.arctan2(Q_data1[0], I_data1[0]) - np.arctan2(Q_data2[0], I_data2[0])
    phase80 = np.arctan2(Q_data1[80], I_data1[80]) - np.arctan2(Q_data2[80], I_data2[80])

    if phase80 < phase0:
        phase80 = phase80 + 2*np.pi

    phase_diff = phase80 - phase0
    if phase_diff < 0:
        phase_diff = phase_diff + 2*np.pi
    phase_diff_list.append(phase_diff)
    phase_0_list.append(phase0)

plt.hist(phase_diff_list,100)
plt.xlabel('phase change in 80 us')
plt.ylabel('count')
plt.savefig('wired result.png')
plt.show()

plt.plot(phase_0_list)
plt.show()

6
I_data1 = data1['I_data'][index]
Q_data1 = data1['Q_data'][index]

I_data2 = data2['I_data'][index]
Q_data2 = data2['Q_data'][index]

complex_data1 = I_data1 + 1j * Q_data1
complex_data2 = I_data2 + 1j * Q_data2

# 计算 FFT
fft_result1 = np.fft.fft(complex_data1)
fft_result2 = np.fft.fft(complex_data2)


# 可视化结果（幅度谱）
frequencies1 = np.fft.fftfreq(len(complex_data1), d=1e-6)  # 1微秒的采样间隔
frequencies2 = np.fft.fftfreq(len(complex_data2), d=1e-6)

plt.plot(frequencies1, np.abs(fft_result1))
plt.plot(frequencies2, np.abs(fft_result2))
plt.title('FFT Result (Magnitude Spectrum)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.xlim(0, max(frequencies1) // 2)  # 只显示非负频率部分
plt.show()