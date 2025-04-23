import numpy as np
import matplotlib.pyplot as plt

# Constants
f = 2.4e9  # Frequency in Hz
t = np.linspace(0, 8e-4, 1000000)  # Time from 0 to 10 microseconds

# Frequencies for the cosine functions
f1 = 1.000001 * f
f2 = 1.000004 * f

# Initial phases
phi1_initial = np.pi / 2
phi2_initial = np.pi / 3

# Cosine functions
x1 = np.cos(2 * np.pi * f1 * t + phi1_initial)
x2 = np.cos(2 * np.pi * f2 * t + phi2_initial)

# Phase difference
phase_diff = (2 * np.pi * f1 * t + phi1_initial) - (2 * np.pi * f2 * t + phi2_initial)
for i in range(len(phase_diff)):
    while phase_diff[i] > np.pi:
        phase_diff[i] = phase_diff[i] - 2*np.pi
    while phase_diff[i] < -np.pi:
        phase_diff[i] = phase_diff[i] + 2*np.pi
# Plotting
plt.figure(figsize=(12, 6))

# Plot cosine functions
plt.subplot(2, 1, 1)
plt.plot(t, x1, color='blue')
plt.plot(t, x2, color='orange')
plt.title('Cosine Functions')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Plot phase difference
plt.subplot(2, 1, 2)
plt.plot(t, phase_diff, label='Phase Difference', color='green')
plt.title('Phase Difference Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Phase Difference (radians)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()