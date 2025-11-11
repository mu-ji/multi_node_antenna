"""
离线数据处理脚本 - 数据读取
用于读取保存在 three_rx_experiment 文件夹下的实验数据
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


import cmath

SPEED_OF_LIGHT  = 299792458

def load_data(filename):
    """
    加载保存的实验数据
    
    参数:
        filename: 数据文件路径
    
    返回:
        data_dict: 包含所有数据的字典
            - angle_list: 角度列表
            - num_triggers: 触发次数
            - triggers: 每次触发的数据列表
                - index: 触发索引
                - angle: 角度
                - rx1_data: 接收端1的数据
                - rx2_data: 接收端2的数据
                - rx3_data: 接收端3的数据
    """
    data = np.load(filename, allow_pickle=True)
    
    # 提取基本信息
    angle_list = data['angle_list']
    num_triggers = int(data['num_triggers'])
    
    # 组织每次触发的数据
    triggers = []
    for idx in range(num_triggers):
        prefix = f'trigger_{idx:03d}'
        trigger_data = {
            'index': idx,
            'angle': float(data[f'{prefix}_angle']),
            'rx1_data': {
                'interval': int(data[f'{prefix}_rx1_interval']),
                'tx1_sqn': int(data[f'{prefix}_rx1_tx1_sqn']),
                'tx2_sqn': int(data[f'{prefix}_rx1_tx2_sqn']),
                'packet_1_I_data': data[f'{prefix}_rx1_pkt1_I'],
                'packet_1_Q_data': data[f'{prefix}_rx1_pkt1_Q'],
                'packet_2_I_data': data[f'{prefix}_rx1_pkt2_I'],
                'packet_2_Q_data': data[f'{prefix}_rx1_pkt2_Q'],
            },
            'rx2_data': {
                'interval': int(data[f'{prefix}_rx2_interval']),
                'tx1_sqn': int(data[f'{prefix}_rx2_tx1_sqn']),
                'tx2_sqn': int(data[f'{prefix}_rx2_tx2_sqn']),
                'packet_1_I_data': data[f'{prefix}_rx2_pkt1_I'],
                'packet_1_Q_data': data[f'{prefix}_rx2_pkt1_Q'],
                'packet_2_I_data': data[f'{prefix}_rx2_pkt2_I'],
                'packet_2_Q_data': data[f'{prefix}_rx2_pkt2_Q'],
            },
            'rx3_data': {
                'interval': int(data[f'{prefix}_rx3_interval']),
                'tx1_sqn': int(data[f'{prefix}_rx3_tx1_sqn']),
                'tx2_sqn': int(data[f'{prefix}_rx3_tx2_sqn']),
                'packet_1_I_data': data[f'{prefix}_rx3_pkt1_I'],
                'packet_1_Q_data': data[f'{prefix}_rx3_pkt1_Q'],
                'packet_2_I_data': data[f'{prefix}_rx3_pkt2_I'],
                'packet_2_Q_data': data[f'{prefix}_rx3_pkt2_Q'],
            }
        }
        triggers.append(trigger_data)
    
    return {
        'angle_list': angle_list,
        'num_triggers': num_triggers,
        'triggers': triggers
    }


def show_slotframe(trigger_data, save_path=None):
    """
    可视化某一次触发中所有相位的变化
    
    参数:
        trigger_data: 单次触发的数据字典，包含 rx1_data, rx2_data, rx3_data
        save_path: 可选，保存图像的路径
    
    返回:
        无，直接显示或保存图像
    """
    # 提取三个接收端的数据
    rx1_data = trigger_data['rx1_data']
    rx2_data = trigger_data['rx2_data']
    rx3_data = trigger_data['rx3_data']
    
    # 计算每个接收端每个包的相位
    # RX1
    rx1_pkt1_phase = np.arctan2(rx1_data['packet_1_Q_data'], rx1_data['packet_1_I_data'])
    rx1_pkt2_phase = np.arctan2(rx1_data['packet_2_Q_data'], rx1_data['packet_2_I_data'])
    
    # RX2
    rx2_pkt1_phase = np.arctan2(rx2_data['packet_1_Q_data'], rx2_data['packet_1_I_data'])
    rx2_pkt2_phase = np.arctan2(rx2_data['packet_2_Q_data'], rx2_data['packet_2_I_data'])
    
    # RX3
    rx3_pkt1_phase = np.arctan2(rx3_data['packet_1_Q_data'], rx3_data['packet_1_I_data'])
    rx3_pkt2_phase = np.arctan2(rx3_data['packet_2_Q_data'], rx3_data['packet_2_I_data'])
    
    # 获取数据长度
    num_samples = len(rx1_pkt1_phase)
    sample_indices = np.arange(num_samples)
    
    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Phase Changes - Trigger {trigger_data["index"]} (Angle: {trigger_data["angle"]:.2f}°)', 
                 fontsize=14, fontweight='bold')
    
    # RX1 Packet 1
    ax1 = axes[0, 0]
    ax1.plot(sample_indices, rx1_pkt1_phase, 'b-o', markersize=3, linewidth=1.5, label='RX1 Pkt1')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_title('RX1 Packet 1 Phase')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-np.pi, np.pi])
    
    # RX1 Packet 2
    ax2 = axes[0, 1]
    ax2.plot(sample_indices, rx1_pkt2_phase, 'b-s', markersize=3, linewidth=1.5, label='RX1 Pkt2')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_title('RX1 Packet 2 Phase')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-np.pi, np.pi])
    
    # RX2 Packet 1
    ax3 = axes[1, 0]
    ax3.plot(sample_indices, rx2_pkt1_phase, 'g-o', markersize=3, linewidth=1.5, label='RX2 Pkt1')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('RX2 Packet 1 Phase')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-np.pi, np.pi])
    
    # RX2 Packet 2
    ax4 = axes[1, 1]
    ax4.plot(sample_indices, rx2_pkt2_phase, 'g-s', markersize=3, linewidth=1.5, label='RX2 Pkt2')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('RX2 Packet 2 Phase')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([-np.pi, np.pi])
    
    # RX3 Packet 1
    ax5 = axes[2, 0]
    ax5.plot(sample_indices, rx3_pkt1_phase, 'r-o', markersize=3, linewidth=1.5, label='RX3 Pkt1')
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_title('RX3 Packet 1 Phase')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim([-np.pi, np.pi])
    
    # RX3 Packet 2
    ax6 = axes[2, 1]
    ax6.plot(sample_indices, rx3_pkt2_phase, 'r-s', markersize=3, linewidth=1.5, label='RX3 Pkt2')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_title('RX3 Packet 2 Phase')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_ylim([-np.pi, np.pi])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show() 

def calculate_angle(I1, Q1, I2, Q2):
    """
    计算第一个IQ向量到第二个IQ向量之间的夹角
    
    参数:
        I1, Q1: 第一个IQ向量的I和Q分量（可以是标量或数组）
        I2, Q2: 第二个IQ向量的I和Q分量（可以是标量或数组）
    
    返回:
        theta: 夹角（弧度），范围 [-π, π]
               如果输入是数组，返回对应位置的夹角数组
    """
    # 转换为numpy数组以确保兼容性
    I1 = np.asarray(I1)
    Q1 = np.asarray(Q1)
    I2 = np.asarray(I2)
    Q2 = np.asarray(Q2)
    
    # 计算两个向量的相位
    phase1 = np.arctan2(Q1, I1)  # 第一个向量的相位
    phase2 = np.arctan2(Q2, I2)  # 第二个向量的相位
    
    # 计算相位差（从phase1到phase2）
    theta = phase1 - phase2
    
    # 将角度约束到 [-π, π] 范围
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    return theta

def cal_slope(phase_diff):
    x = np.arange(len(phase_diff)).reshape(-1, 1)
    y = phase_diff
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_

def phase_to_angle(geo):
    val = geo/(2*np.pi)*12.5/6.3
    if val > 1:
        return 90
    if val < -1:
        return -90
    return np.arcsin(val)/np.pi*180
    
def compensate_phase_offset(trigger_data, ref_position):
    # 提取三个接收端的数据
    rx1_data = trigger_data['rx1_data']
    rx2_data = trigger_data['rx2_data']
    rx3_data = trigger_data['rx3_data']
    
    # 计算每个接收端每个包的相位
    # RX1
    rx1_pkt1_phase = np.arctan2(rx1_data['packet_1_Q_data'], rx1_data['packet_1_I_data'])
    rx1_pkt2_phase = np.arctan2(rx1_data['packet_2_Q_data'], rx1_data['packet_2_I_data'])
    
    # RX2
    rx2_pkt1_phase = np.arctan2(rx2_data['packet_1_Q_data'], rx2_data['packet_1_I_data'])
    rx2_pkt2_phase = np.arctan2(rx2_data['packet_2_Q_data'], rx2_data['packet_2_I_data'])
    
    # RX3
    rx3_pkt1_phase = np.arctan2(rx3_data['packet_1_Q_data'], rx3_data['packet_1_I_data'])
    rx3_pkt2_phase = np.arctan2(rx3_data['packet_2_Q_data'], rx3_data['packet_2_I_data'])

    interval = (rx1_data['interval'] + rx2_data['interval'] + rx3_data['interval'])/3
    # 计算每个接收端每个包的相位
    # RX1

    pkt1_phase_diff_12 = calculate_angle(rx1_data['packet_1_I_data'], rx1_data['packet_1_Q_data'], rx2_data['packet_1_I_data'], rx2_data['packet_1_Q_data'])
    pkt1_phase_diff_23 = calculate_angle(rx2_data['packet_1_I_data'], rx2_data['packet_1_Q_data'], rx3_data['packet_1_I_data'], rx3_data['packet_1_Q_data'])

    pkt2_phase_diff_12 = calculate_angle(rx1_data['packet_2_I_data'], rx1_data['packet_2_Q_data'], rx2_data['packet_2_I_data'], rx2_data['packet_2_Q_data'])
    pkt2_phase_diff_23 = calculate_angle(rx2_data['packet_2_I_data'], rx2_data['packet_2_Q_data'], rx3_data['packet_2_I_data'], rx3_data['packet_2_Q_data'])

    pkt1_phase_diff_12 = np.unwrap(pkt1_phase_diff_12)
    pkt1_phase_diff_23 = np.unwrap(pkt1_phase_diff_23)
    pkt2_phase_diff_12 = np.unwrap(pkt2_phase_diff_12)
    pkt2_phase_diff_23 = np.unwrap(pkt2_phase_diff_23)

    slope_12, intercept_12 = cal_slope(pkt1_phase_diff_12)
    slope_23, intercept_23 = cal_slope(pkt1_phase_diff_23)  
    print(slope_12, slope_23)

    target_phase_diff_12 = np.arctan2(np.sin(pkt2_phase_diff_12 - pkt1_phase_diff_12 + slope_12 * interval/16), np.cos(pkt2_phase_diff_12 - pkt1_phase_diff_12 + slope_12 * interval/16))
    target_phase_diff_23 = np.arctan2(np.sin(pkt2_phase_diff_23 - pkt1_phase_diff_23 + slope_23 * interval/16), np.cos(pkt2_phase_diff_23 - pkt1_phase_diff_23 + slope_23 * interval/16))

    target_phase_diff_12_unwrap = np.unwrap(target_phase_diff_12)
    target_phase_diff_23_unwrap = np.unwrap(target_phase_diff_23)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # axes[0].plot(target_phase_diff_12, marker='.')
    # axes[0].plot(target_phase_diff_12_unwrap, marker='.')
    # axes[0].set_ylim([-np.pi, np.pi])
    # axes[0].legend()

    # axes[1].plot(target_phase_diff_13, marker='.')
    # axes[1].plot(target_phase_diff_13_unwrap, marker='.')
    # axes[1].set_ylim([-np.pi, np.pi])
    # axes[1].legend()

    # plt.tight_layout()
    # plt.legend()
    # plt.show()

    # fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    # ax1 = axes[0, 0]
    # ax1.plot(rx1_pkt1_phase, 'b-o', markersize=3, linewidth=1.5, label='RX1 Pkt1')
    # ax1.set_xlabel('Sample Index')
    # ax1.set_ylabel('Phase (rad)')
    # ax1.set_title('RX1 Packet 1 Phase')
    # ax1.grid(True, alpha=0.3)
    # ax1.legend()
    # ax1.set_ylim([-np.pi, np.pi])
    
    # # RX1 Packet 2
    # ax2 = axes[0, 1]
    # ax2.plot(rx1_pkt2_phase, 'b-s', markersize=3, linewidth=1.5, label='RX1 Pkt2')
    # ax2.set_xlabel('Sample Index')
    # ax2.set_ylabel('Phase (rad)')
    # ax2.set_title('RX1 Packet 2 Phase')
    # ax2.grid(True, alpha=0.3)
    # ax2.legend()
    # ax2.set_ylim([-np.pi, np.pi])
    
    # # RX2 Packet 1
    # ax3 = axes[1, 0]
    # ax3.plot(rx2_pkt1_phase, 'g-o', markersize=3, linewidth=1.5, label='RX2 Pkt1')
    # ax3.set_xlabel('Sample Index')
    # ax3.set_ylabel('Phase (rad)')
    # ax3.set_title('RX2 Packet 1 Phase')
    # ax3.grid(True, alpha=0.3)
    # ax3.legend()
    # ax3.set_ylim([-np.pi, np.pi])
    
    # # RX2 Packet 2
    # ax4 = axes[1, 1]
    # ax4.plot(rx2_pkt2_phase, 'g-s', markersize=3, linewidth=1.5, label='RX2 Pkt2')
    # ax4.set_xlabel('Sample Index')
    # ax4.set_ylabel('Phase (rad)')
    # ax4.set_title('RX2 Packet 2 Phase')
    # ax4.grid(True, alpha=0.3)
    # ax4.legend()
    # ax4.set_ylim([-np.pi, np.pi])
    
    # # RX3 Packet 1
    # ax5 = axes[2, 0]
    # ax5.plot(rx3_pkt1_phase, 'r-o', markersize=3, linewidth=1.5, label='RX3 Pkt1')
    # ax5.set_xlabel('Sample Index')
    # ax5.set_ylabel('Phase (rad)')
    # ax5.set_title('RX3 Packet 1 Phase')
    # ax5.grid(True, alpha=0.3)
    # ax5.legend()
    # ax5.set_ylim([-np.pi, np.pi])
    
    # # RX3 Packet 2
    # ax6 = axes[2, 1]
    # ax6.plot(rx3_pkt2_phase, 'r-s', markersize=3, linewidth=1.5, label='RX3 Pkt2')
    # ax6.set_xlabel('Sample Index')
    # ax6.set_ylabel('Phase (rad)')
    # ax6.set_title('RX3 Packet 2 Phase')
    # ax6.grid(True, alpha=0.3)
    # ax6.legend()
    # ax6.set_ylim([-np.pi, np.pi])
    
    # axes[0, 2].plot(pkt1_phase_diff_12, marker='.')
    # axes[0, 2].set_ylim([-np.pi, np.pi])
    # axes[0, 2].legend()

    # axes[0, 3].plot(pkt1_phase_diff_23, marker='.')
    # axes[0, 3].set_ylim([-np.pi, np.pi])
    # axes[0, 3].legend()

    # axes[1, 2].plot(pkt2_phase_diff_12, marker='.')
    # axes[1, 2].set_ylim([-np.pi, np.pi])
    # axes[1, 2].legend()

    # axes[1, 3].plot(pkt2_phase_diff_23, marker='.')
    # axes[1, 3].set_ylim([-np.pi, np.pi])
    # axes[1, 3].legend()


    # axes[2, 2].plot(target_phase_diff_12_unwrap, marker='.')
    # axes[2, 2].set_ylim([-np.pi, np.pi])
    # axes[2, 2].legend()

    # axes[2, 3].plot(target_phase_diff_23_unwrap, marker='.')
    # axes[2, 3].set_ylim([-np.pi, np.pi])
    # axes[2, 3].legend()

    # plt.tight_layout()
    # plt.show()

    phase_12 = np.mean(target_phase_diff_12_unwrap)
    phase_23 = np.mean(target_phase_diff_23_unwrap)



    ant1_theta = phase_12
    ant2_theta = phase_12 + phase_23
    ant0_theta = 0

    def steering_vector(alpha):
        j = 1j  # 复数单位
        return np.array([1, cmath.exp(-j * 2 * np.pi * 2.4e9 * (0.063*np.sin(alpha)/SPEED_OF_LIGHT)), cmath.exp(-j * 2 * np.pi * 2.4e9 * 2*(0.063*np.sin(alpha)/SPEED_OF_LIGHT))])

    received_signal = np.array([cmath.exp(1j*ant0_theta), cmath.exp(1j*ant1_theta), cmath.exp(1j*ant2_theta)])
    print(received_signal)
    angle_list = [np.radians(i) for i in range(-90, 90)]
    y_alpha_list = []
    for alpha in angle_list:
        y_alpha = steering_vector(alpha)[0]*received_signal[0] + steering_vector(alpha)[1]*received_signal[1] + steering_vector(alpha)[2]*received_signal[2]
        y_alpha_list.append(y_alpha)

    print([i for i in range(-90, 90)][np.argmax(np.array(y_alpha_list))])
    return [i for i in range(-90, 90)][np.argmax(np.array(y_alpha_list))], phase_to_angle(phase_12), phase_to_angle(phase_23)

# 使用示例：
if __name__ == '__main__':
    # 指定要读取的数据文件名
    filename = 'three_rx_experiment/angle_-45.npz'
    
    # 加载数据
    data_dict = load_data(filename)
    print(f"成功加载 {data_dict['num_triggers']} 次触发数据")
    print(f"角度列表: {data_dict['angle_list']}")
    
    est_angle_list = []
    phase12_list = []
    phase23_list = []

    for i in range(len(data_dict['triggers'])):
        trigger = data_dict['triggers'][i]
        # show_slotframe(trigger)
        angle,phase12,phase23 = compensate_phase_offset(trigger, 0)
        est_angle_list.append(angle)
        phase12_list.append(phase12)
        phase23_list.append(phase23)
    plt.figure()
    plt.boxplot(phase12_list, positions=[1])
    plt.boxplot(phase23_list, positions=[2])
    plt.boxplot(est_angle_list, positions=[3])
    plt.show()

    plt.figure()
    vp12 = plt.violinplot(phase12_list, positions=[1])
    for body in vp12['bodies']:
        body.set_label('rx12 angle')
    vp23 = plt.violinplot(phase23_list, positions=[2])
    for body in vp23['bodies']:
        body.set_label('rx23 angle')
    vp123 = plt.violinplot(est_angle_list, positions=[3])
    for body in vp123['bodies']:
        body.set_label('rx123 angle')
    plt.legend()
    plt.show()

    # plt.scatter([i for i in range(len(est_angle_list))], est_angle_list)
    # plt.show()