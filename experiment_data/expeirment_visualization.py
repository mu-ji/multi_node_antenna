import numpy as np
import matplotlib.pyplot as plt

est_angle_list = np.load('experiment_data/angle_10.npz')['arr_0']
angle_list = [-45, -30, -10, 0, 10, 30, 45]

plt.figure()

for i in angle_list:
    est_angle_list = np.load('experiment_data/angle_%d.npz'%i)['arr_0']
    plt.violinplot(est_angle_list, positions=[i], widths=6)

plt.xticks(angle_list, [f'{i}°' for i in angle_list])
plt.xlim(-60, 60)
plt.ylim(-90, 100)
plt.xlabel('True Angle (°)')
plt.ylabel('Estimated Angle')
plt.title('Angle Estimation Distribution')
plt.grid(True, alpha=0.3)
plt.show()