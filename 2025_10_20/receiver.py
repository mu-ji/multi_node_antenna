import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii
import threading
from collections import defaultdict, deque
import time

from scipy import stats
import itertools

ser1 = serial.Serial('COM14', 115200)
ser2 = serial.Serial('COM16', 115200)

SPEED_OF_LIGHT = 299792458


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
    print("=" * 50)
    print("正斜率最优解:")
    print(f"  最佳 R²: {best_positive['r_squared']:.6f}")
    print(f"  最佳补偿倍数: {best_positive['k_values']}")
    print(f"  最佳斜率: {best_positive['slope']:.6f}")
    
    print("\n负斜率最优解:")
    print(f"  最佳 R²: {best_negative['r_squared']:.6f}")
    print(f"  最佳补偿倍数: {best_negative['k_values']}")
    print(f"  最佳斜率: {best_negative['slope']:.6f}")
    print("=" * 50)
    
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

def warp_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
def estimate_angle(timestamps, phase_diff):
    (is_linear_pos, is_linear_neg, is_linear_overall, 
    best_pos, best_neg, best_overall, direction) = check_linearity_with_phase_unwrap_dual(
        timestamps[:4], phase_diff[:4], max_k=2, r_squared_threshold=0.98
    )
    pos_new_phase_list = []
    for i in range(5):
        pos_new_phase_list.append(phase_diff[i] + best_pos['k_values'][i] * 2 * np.pi)

    neg_new_phase_list = []
    for i in range(5):
        neg_new_phase_list.append(phase_diff[i] + best_neg['k_values'][i] * 2 * np.pi)

    timestamps_init = timestamps[0]
    timestamps = [timestamps[i]-timestamps_init for i in range(len(timestamps))]

    pos_intercept = np.mean(pos_new_phase_list) - best_pos['slope'] * np.mean(timestamps)
    neg_intercept = np.mean(neg_new_phase_list) - best_neg['slope'] * np.mean(timestamps)

    pos_phase_delta = pos_intercept + best_pos['slope'] * (timestamps[4])
    neg_phase_delta = neg_intercept + best_neg['slope'] * (timestamps[4])

    pos_phase_est = warp_to_pi(pos_phase_delta)
    neg_phase_est = warp_to_pi(neg_phase_delta)

    pos_phase_delta_tx2 = pos_intercept + best_pos['slope'] * (timestamps[5])
    neg_phase_delta_tx2 = neg_intercept + best_neg['slope'] * (timestamps[5])

    pos_phase_est_tx2 = warp_to_pi(pos_phase_delta_tx2)
    neg_phase_est_tx2 = warp_to_pi(neg_phase_delta_tx2)

    plt.figure()
    plt.plot(timestamps, pos_new_phase_list, 'o-', label='Pos Compensated Phase')
    plt.plot(timestamps, neg_new_phase_list, 'o-', label='Neg Compensated Phase')
    plt.plot(timestamps, phase_diff, 'x-', label='Original Phase')
    # plt.plot(timestamps, np.arctan2(rx1_pkt_array[5]['Q_data'][0], rx1_pkt_array[5]['I_data'][0]) - np.arctan2(rx2_pkt_array[5]['Q_data'][0], rx2_pkt_array[5]['I_data'][0]), 'x-', c='g')
    # plt.plot(rx1_pkt_array[5]['timestamp'] - rx1_timestamp_init, pos_tx2_phase_delta, 'o-', c='b')
    # plt.plot(rx1_pkt_array[5]['timestamp'] - rx1_timestamp_init, neg_tx2_phase_delta, 'o-', c='orange')
    plt.legend()
    plt.show()

    if abs(pos_phase_est - phase_diff[4]) < 0.05:
        pos_tx2_phase_diff = phase_diff[5] - pos_phase_est_tx2 
        pos_tx2_phase_diff = pos_tx2_phase_diff/6.28*12.5/6
        print('pos_angle1:', np.arcsin(pos_tx2_phase_diff)/np.pi*180)

    if abs(neg_phase_est - phase_diff[4]) < 0.05:
        neg_tx2_phase_diff = phase_diff[5] - neg_phase_est_tx2
        neg_tx2_phase_diff = neg_tx2_phase_diff/6.28*12.5/6
        print('neg_angle1:', np.arcsin(neg_tx2_phase_diff)/np.pi*180)

    if abs(pos_phase_est - phase_diff[4]) > 0.05 and abs(neg_phase_est - phase_diff[4]) > 0.05:
        print('no good frame')

    return True

# 全局变量用于线程间通信
shared_data = {
    'ser1_data': defaultdict(list),  # 存储每个pkt_sqn的完整数据
    'ser2_data': defaultdict(list),
    'ser1_phase_data': defaultdict(list),  # 新增：存储相位数据
    'ser2_phase_data': defaultdict(list),
    'lock': threading.Lock(),
    'trigger_count': 0,
    'last_triggered_pkt_sqn': None,
    'max_triggers': 10,
    'triggered_pkts': set()
}

def parse_phase_data(raw_data):
    """解析相位数据"""
    # 前4个字节是IQ数据（2个I + 2个Q）
    i_data = struct.unpack('>h', bytes(raw_data[2:4]))[0] 
    q_data = struct.unpack('>h', bytes(raw_data[0:2]))[0]  
    
    # 计算相位
    phase = math.atan2(q_data, i_data)
    
    return phase, i_data, q_data


def process_phase_data(pkt_sqn):
    """处理相位数据的函数"""
    with shared_data['lock']:
        ser1_phases = shared_data['ser1_phase_data'].get(pkt_sqn, [])
        ser2_phases = shared_data['ser2_phase_data'].get(pkt_sqn, [])
    
    # 按inner_sqn排序
    ser1_phases_sorted = sorted(ser1_phases, key=lambda x: x['inner_sqn'])
    ser2_phases_sorted = sorted(ser2_phases, key=lambda x: x['inner_sqn'])
    
    print(f"\n=== 处理pkt_sqn {pkt_sqn} 的相位数据 ===")
    # print(f"串口1数据: {len(ser1_phases_sorted)} 个相位点")
    # print(f"串口2数据: {len(ser2_phases_sorted)} 个相位点")
    
    # # 打印详细数据
    # for i, (phase1, phase2) in enumerate(zip(ser1_phases_sorted, ser2_phases_sorted)):
    #     print(f"inner_sqn {i}: 串口1相位={phase1['phase']:.4f}, 串口2相位={phase2['phase']:.4f}")

    phase_diff = [phase1['phase'] - phase2['phase'] for phase1, phase2 in zip(ser1_phases_sorted, ser2_phases_sorted)]
    timestamps = [phase1['timestamp'] for phase1 in ser1_phases_sorted]
    estimate_angle(phase_diff, timestamps)
    # ser1_phase = [ser1_phases_sorted[i]['phase'] for i in len(ser1_phases_sorted)]
    # print(ser1_phase)

    print(f"\n===   {pkt_sqn} 的相位数据 处理完成===")
    return ser1_phases_sorted, ser2_phases_sorted


def thread(ser, ser_id):
    rawFrame = []
    last_cleanup = time.time()
    
    while True:
        byte = ser.read(1)
        if byte:
            rawFrame.append(byte[0])
            
            if len(rawFrame) >= 4 and rawFrame[-4:] == [255, 255, 255, 255]:
                if len(rawFrame) == 14:
                    # 解析数据
                    received_timestamp = struct.unpack('>I', bytes(rawFrame[4:8]))[0]
                    received_pkt_sqn = rawFrame[8]
                    received_pkt_inner_sqn = rawFrame[9]
                    
                    # 解析相位数据
                    phase, i_data, q_data = parse_phase_data(rawFrame[0:4])
                    
                    phase_data = {
                        'phase': phase,
                        'i_data': i_data,
                        'q_data': q_data,
                        'timestamp': received_timestamp
                    }
                    
                    print(f"串口{ser_id} - pkt_sqn: {received_pkt_sqn}, inner_sqn: {received_pkt_inner_sqn}, "
                            f"相位: {phase:.4f}, I: {i_data}, Q: {q_data}")
                    
                    rawFrame = []
                else:
                    rawFrame = []
                        

def start_monitoring(ser1, ser2):
    print("串口监控已启动...")
    print(f"目标：触发 {shared_data['max_triggers']} 次函数")
    
    thread1 = threading.Thread(target=thread, args=(ser1, 1))
    thread2 = threading.Thread(target=thread, args=(ser2, 2))
    thread1.daemon = True
    thread2.daemon = True
    
    thread1.start()
    thread2.start()
    
    return thread1, thread2

# 启动监控
start_monitoring(ser1, ser2)

