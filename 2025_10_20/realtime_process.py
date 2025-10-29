import threading
import time
import struct
from collections import defaultdict
import numpy as np
import serial
from scipy import stats
import itertools
import matplotlib.pyplot as plt 

ser1_data = {
    'timestamp': [-1]*6,
    'phase': [-1]*6,
    'pkt_sqn': [-1]*6
}

ser2_data = {
    'timestamp': [-1]*6,
    'phase': [-1]*6,
    'pkt_sqn': [-1]*6
}

def check_linearity_with_phase_unwrap_dual(timestamps, phases, max_k=6, r_squared_threshold=0.98):
    """
    检查在2π补偿下是否存在线性关系，分别拟合正负斜率的最优解
    max_k: 每个点最多补偿的2π倍数范围 [-max_k, max_k]
    """
    n_points = len(phases)
    
    # 生成所有可能的补偿组合
    k_combinations = list(itertools.product(range(-max_k, max_k+1), repeat=n_points))
    
    # 分别存储正负斜率的最优解
    best_positive = {
        'r_squared': -1,
        'k_values': None,
        'slope': None,
        'intercept': None
    }
    
    best_negative = {
        'r_squared': -1,
        'k_values': None,
        'slope': None,
        'intercept': None
    }
    
    for k_vec in k_combinations:
        # 应用补偿
        compensated_phases = phases + np.array(k_vec) * 2 * np.pi
        
        # 线性拟合
        if len(np.unique(timestamps)) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, compensated_phases)
            r_squared = r_value ** 2
            
            # 根据斜率正负分别更新最优解
            if slope >= 0:  # 正斜率
                if r_squared > best_positive['r_squared']:
                    best_positive['r_squared'] = r_squared
                    best_positive['k_values'] = k_vec
                    best_positive['slope'] = slope
                    best_positive['intercept'] = intercept
            else:  # 负斜率
                if r_squared > best_negative['r_squared']:
                    best_negative['r_squared'] = r_squared
                    best_negative['k_values'] = k_vec
                    best_negative['slope'] = slope
                    best_negative['intercept'] = intercept
    
    # 输出结果
    # print("=" * 50)
    # print("正斜率最优解:")
    # print(f"  最佳 R²: {best_positive['r_squared']:.6f}")
    # print(f"  最佳补偿倍数: {best_positive['k_values']}")
    # print(f"  最佳斜率: {best_positive['slope']:.6f}")
    
    # print("\n负斜率最优解:")
    # print(f"  最佳 R²: {best_negative['r_squared']:.6f}")
    # print(f"  最佳补偿倍数: {best_negative['k_values']}")
    # print(f"  最佳斜率: {best_negative['slope']:.6f}")
    # print("=" * 50)
    
    # 判断哪个方向的线性关系更好
    if best_positive['r_squared'] > best_negative['r_squared']:
        best_direction = 'positive'
        best_overall = best_positive
    else:
        best_direction = 'negative'
        best_overall = best_negative
    
    print(f"总体最优方向: {best_direction} (R² = {best_overall['r_squared']:.6f})")
    
    # 判断是否存在强线性关系
    is_linear_positive = best_positive['r_squared'] > r_squared_threshold
    is_linear_negative = best_negative['r_squared'] > r_squared_threshold
    is_linear_overall = best_overall['r_squared'] > r_squared_threshold
    
    return (is_linear_positive, is_linear_negative, is_linear_overall, 
            best_positive, best_negative, best_overall, best_direction)


def thread(ser, ser_id):
    rawFrame = []
    
    while True:
        byte = ser.read(1)
        if byte:
            rawFrame.append(byte[0])
            if len(rawFrame) >= 4 and rawFrame[-4:] == [255, 255, 255, 255]:
                if len(rawFrame) == 14:
                    received_timestamp = struct.unpack('>I', bytes(rawFrame[4:8]))[0]
                    received_pkt_sqn = rawFrame[8]
                    received_pkt_inner_sqn = rawFrame[9]

                    i_data = struct.unpack('>h', bytes(rawFrame[2:4]))[0] 
                    q_data = struct.unpack('>h', bytes(rawFrame[0:2]))[0]  
    
                    # 计算相位
                    phase = np.arctan2(q_data, i_data)
                    
                    if ser_id == 1:
                        ser1_data['timestamp'][received_pkt_inner_sqn] = received_timestamp
                        ser1_data['phase'][received_pkt_inner_sqn] = phase
                        ser1_data['pkt_sqn'][received_pkt_inner_sqn] = received_pkt_sqn
                    elif ser_id == 2:
                        ser2_data['timestamp'][received_pkt_inner_sqn] = received_timestamp
                        ser2_data['phase'][received_pkt_inner_sqn] = phase
                        ser2_data['pkt_sqn'][received_pkt_inner_sqn] = received_pkt_sqn

                    # print(f"串口{ser_id} - pkt_sqn: {received_pkt_sqn}, inner_sqn: {received_pkt_inner_sqn}")
                    
                    # 接收到inner_sqn=5时终止线程
                    if received_pkt_inner_sqn == 5:
                        break
                    
                rawFrame = []

def check_pkt_sqn(list1, list2):
    # 检查每个list内部只有一个唯一元素，且两个list相同
    return (list1 == list2 and 
            len(set(list1)) <= 1 and 
            len(set(list2)) <= 1)
def check_data_completeness_and_validity(ser1_data, ser2_data):
    if check_pkt_sqn(ser1_data['pkt_sqn'], ser2_data['pkt_sqn']):
        return True
    else:
        return False

def warp_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def estimate_next_phase(timestamps, phase_diff):

    num = len(timestamps)

    (is_linear_pos, is_linear_neg, is_linear_overall, best_pos, best_neg, best_overall, direction) = check_linearity_with_phase_unwrap_dual(
        timestamps, phase_diff, max_k=2, r_squared_threshold=0.98)
    
    pos_new_phase_list = []
    for i in range(num):
        pos_new_phase_list.append(phase_diff[i] + best_pos['k_values'][i] * 2 * np.pi)

    neg_new_phase_list = []
    for i in range(num):
        neg_new_phase_list.append(phase_diff[i] + best_neg['k_values'][i] * 2 * np.pi)

    timestamps_init = timestamps[0]
    timestamps = [timestamps[i]-timestamps_init for i in range(len(timestamps))]

    pos_intercept = np.mean(pos_new_phase_list) - best_pos['slope'] * np.mean(timestamps)
    neg_intercept = np.mean(neg_new_phase_list) - best_neg['slope'] * np.mean(timestamps)

    pos_phase_delta = pos_intercept + best_pos['slope'] * (timestamps[num-1])
    neg_phase_delta = neg_intercept + best_neg['slope'] * (timestamps[num-1])

    pos_phase_est = warp_to_pi(pos_phase_delta)
    neg_phase_est = warp_to_pi(neg_phase_delta)

    # plt.figure()
    # plt.plot(timestamps, pos_new_phase_list, 'o-', label='Pos Compensated Phase')
    # plt.plot(timestamps, neg_new_phase_list, 'o-', label='Neg Compensated Phase')
    # plt.plot(timestamps, phase_diff, 'x-', label='Original Phase')
    # # plt.plot(timestamps, np.arctan2(rx1_pkt_array[5]['Q_data'][0], rx1_pkt_array[5]['I_data'][0]) - np.arctan2(rx2_pkt_array[5]['Q_data'][0], rx2_pkt_array[5]['I_data'][0]), 'x-', c='g')
    # # plt.plot(rx1_pkt_array[5]['timestamp'] - rx1_timestamp_init, pos_tx2_phase_delta, 'o-', c='b')
    # # plt.plot(rx1_pkt_array[5]['timestamp'] - rx1_timestamp_init, neg_tx2_phase_delta, 'o-', c='orange')
    # plt.legend()
    # plt.show()

    return pos_phase_est, neg_phase_est
def data_process(ser1_data, ser2_data, threshold, pos_angle_list, neg_angle_list):
    ser1_timestamps = ser1_data['timestamp']
    ser1_phases = ser1_data['phase']
    ser2_timestamps = ser2_data['timestamp']
    ser2_phases = ser2_data['phase']

    phase_diff = np.array(ser1_phases) - np.array(ser2_phases)
    ser1_timestamp_init = ser1_timestamps[0]
    ser2_timestamp_init = ser2_timestamps[0]
    
    ser1_timestamps = np.array([ser1_timestamps[i] - ser1_timestamp_init for i in range(len(ser1_timestamps))])
    ser2_timestamps = np.array([ser2_timestamps[i] - ser2_timestamp_init for i in range(len(ser2_timestamps))])

    # print('-'*50)
    # print(ser1_timestamps)
    # print(ser2_timestamps)
    # print(ser1_timestamps - ser2_timestamps)
    # print('-'*50)

    pos_phase_est, neg_phase_est = estimate_next_phase(ser1_timestamps[:-2], phase_diff[:-2])
    print(pos_phase_est)
    print(neg_phase_est)
    print(phase_diff[-2])

    tx2_pos_est, tx2_neg_est = estimate_next_phase(ser2_timestamps[:-1], phase_diff[:-1])
    pos_angle_est = np.arcsin((phase_diff[-1] - tx2_pos_est)/6.28*12.5/6)/np.pi*180
    neg_angle_est = np.arcsin((phase_diff[-1] - tx2_neg_est)/6.28*12.5/6)/np.pi*180

    if abs(warp_to_pi(pos_phase_est - phase_diff[-2])) < threshold or abs(warp_to_pi(neg_phase_est - phase_diff[-2])) < threshold:
        tx2_pos_est, tx2_neg_est = estimate_next_phase(ser2_timestamps[:-1], phase_diff[:-1])
        pos_angle_est = np.arcsin((phase_diff[-1] - tx2_pos_est)/6.28*12.5/6)/np.pi*180
        neg_angle_est = np.arcsin((phase_diff[-1] - tx2_neg_est)/6.28*12.5/6)/np.pi*180
        pos_angle_list.append(pos_angle_est)
        neg_angle_list.append(neg_angle_est)
    else:
        print('no good frame')


    return 0

def start_monitoring(ser1, ser2):
    global ser1_data, ser2_data
    print("串口监控已启动...")
    
    pos_angle_list = []
    neg_angle_list = []

    while len(pos_angle_list)<=50:
        thread1 = threading.Thread(target=thread, args=(ser1, 1))
        thread2 = threading.Thread(target=thread, args=(ser2, 2))
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        
        # 等待两个线程终止
        thread1.join()
        thread2.join()
        
        print("两个线程均已终止，开始检查数据...")
        
        # 检查数据有效性
        is_valid = check_data_completeness_and_validity(ser1_data, ser2_data)
        
        if is_valid:
            # 数据有效，进行处理
            data_process(ser1_data, ser2_data, 0.2, pos_angle_list, neg_angle_list)
        else:
            # 数据无效，清除数据重新开始
            print("数据无效，重新开始监控...")
            ser1_data = {
                'timestamp': [-1]*6,
                'phase': [-1]*6,
                'pkt_sqn': [-1]*6
            }

            ser2_data = {
                'timestamp': [-1]*6,
                'phase': [-1]*6,
                'pkt_sqn': [-1]*6
            }
        
        # 短暂延迟后继续下一轮
        time.sleep(0.1)
    
    print(f"达到最大触发次数{len(pos_angle_list)},监控结束")
    np.savez('2025_10_20/indoor_experiment/angle_{}_pos.npz'.format(0), pos_angle_list)
    np.savez('2025_10_20/indoor_experiment/angle_{}_neg.npz'.format(0), neg_angle_list)

ser1 = serial.Serial('COM14', 115200)
ser2 = serial.Serial('COM16', 115200)
# 启动监控
start_monitoring(ser1, ser2)