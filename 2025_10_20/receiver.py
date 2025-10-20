import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii
import threading
from collections import defaultdict, deque
import time

ser1 = serial.Serial('COM14', 115200)
ser2 = serial.Serial('COM16', 115200)

SPEED_OF_LIGHT = 299792458

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

def check_complete_sequence(ser_id, pkt_sqn, inner_sqn, phase_data):
    """检查是否收到完整的0-5 inner_sqn序列，并保存相位数据"""
    with shared_data['lock']:
        if pkt_sqn in shared_data['triggered_pkts']:
            return False
            
        if ser_id == 1:
            data_dict = shared_data['ser1_data']
            phase_dict = shared_data['ser1_phase_data']
        else:
            data_dict = shared_data['ser2_data']
            phase_dict = shared_data['ser2_phase_data']
        
        # 保存inner_sqn
        if inner_sqn not in data_dict[pkt_sqn]:
            data_dict[pkt_sqn].append(inner_sqn)
        
        # 保存相位数据
        if inner_sqn not in phase_dict[pkt_sqn]:
            phase_dict[pkt_sqn].append({
                'inner_sqn': inner_sqn,
                'phase': phase_data['phase'],
                'i_data': phase_data['i_data'],
                'q_data': phase_data['q_data'],
                'timestamp': phase_data['timestamp']
            })
        
        # 检查是否包含完整的0-5序列
        if len(data_dict[pkt_sqn]) >= 6:
            sorted_seq = sorted(data_dict[pkt_sqn])
            has_complete = all(x in sorted_seq for x in range(6))
            return has_complete
        return False

def check_both_serials_complete(pkt_sqn):
    """检查两个串口是否都收到了同一个pkt_sqn的完整inner_sqn序列"""
    with shared_data['lock']:
        if pkt_sqn in shared_data['triggered_pkts']:
            return False
        
        ser1_has_complete = False
        ser2_has_complete = False
        
        if pkt_sqn in shared_data['ser1_data']:
            ser1_sorted = sorted(shared_data['ser1_data'][pkt_sqn])
            ser1_has_complete = len(ser1_sorted) >= 6 and all(x in ser1_sorted for x in range(6))
        
        if pkt_sqn in shared_data['ser2_data']:
            ser2_sorted = sorted(shared_data['ser2_data'][pkt_sqn])
            ser2_has_complete = len(ser2_sorted) >= 6 and all(x in ser2_sorted for x in range(6))
        
        return ser1_has_complete and ser2_has_complete

def process_phase_data(pkt_sqn):
    """处理相位数据的函数"""
    with shared_data['lock']:
        ser1_phases = shared_data['ser1_phase_data'].get(pkt_sqn, [])
        ser2_phases = shared_data['ser2_phase_data'].get(pkt_sqn, [])
    
    # 按inner_sqn排序
    ser1_phases_sorted = sorted(ser1_phases, key=lambda x: x['inner_sqn'])
    ser2_phases_sorted = sorted(ser2_phases, key=lambda x: x['inner_sqn'])
    
    print(f"\n=== 处理pkt_sqn {pkt_sqn} 的相位数据 ===")
    print(f"串口1数据: {len(ser1_phases_sorted)} 个相位点")
    print(f"串口2数据: {len(ser2_phases_sorted)} 个相位点")
    
    # 打印详细数据
    for i, (phase1, phase2) in enumerate(zip(ser1_phases_sorted, ser2_phases_sorted)):
        print(f"inner_sqn {i}: 串口1相位={phase1['phase']:.4f}, 串口2相位={phase2['phase']:.4f}")

    phase_diff = [phase2['phase'] - phase1['phase'] for phase1, phase2 in zip(ser1_phases_sorted, ser2_phases_sorted)]
    print(phase_diff)
    # ser1_phase = [ser1_phases_sorted[i]['phase'] for i in len(ser1_phases_sorted)]
    # print(ser1_phase)

    print(f"\n===   {pkt_sqn} 的相位数据 处理完成===")
    return ser1_phases_sorted, ser2_phases_sorted

def trigger_function(pkt_sqn):
    """当检测到匹配序列时触发的函数"""
    with shared_data['lock']:
        if pkt_sqn in shared_data['triggered_pkts']:
            return
            
        shared_data['triggered_pkts'].add(pkt_sqn)
        shared_data['trigger_count'] += 1
        shared_data['last_triggered_pkt_sqn'] = pkt_sqn
        
        current_count = shared_data['trigger_count']
        max_triggers = shared_data['max_triggers']
    
    print(f"\n*** 触发函数 #{current_count}: pkt_sqn={pkt_sqn} ***")
    
    # 处理相位数据
    ser1_data, ser2_data = process_phase_data(pkt_sqn)
    
    # 这里可以添加其他处理逻辑
    # 例如：保存数据到文件、更新图形显示等
    
    if current_count >= max_triggers:
        print(f"已达到最大触发次数 {max_triggers}")

def cleanup_old_data():
    """清理旧数据，防止内存泄漏"""
    with shared_data['lock']:
        max_keep_pkts = 20
        
        # 清理ser1数据
        ser1_keys = list(shared_data['ser1_data'].keys())
        if len(ser1_keys) > max_keep_pkts:
            for key in ser1_keys[:-max_keep_pkts]:
                if key in shared_data['ser1_data']:
                    del shared_data['ser1_data'][key]
                if key in shared_data['ser1_phase_data']:
                    del shared_data['ser1_phase_data'][key]
        
        # 清理ser2数据
        ser2_keys = list(shared_data['ser2_data'].keys())
        if len(ser2_keys) > max_keep_pkts:
            for key in ser2_keys[:-max_keep_pkts]:
                if key in shared_data['ser2_data']:
                    del shared_data['ser2_data'][key]
                if key in shared_data['ser2_phase_data']:
                    del shared_data['ser2_phase_data'][key]

def thread(ser, ser_id):
    rawFrame = []
    last_cleanup = time.time()
    
    while True:
        try:
            if time.time() - last_cleanup > 5:
                cleanup_old_data()
                last_cleanup = time.time()
                
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
                        
                        # 检查当前串口是否收到完整序列
                        current_complete = check_complete_sequence(ser_id, received_pkt_sqn, 
                                                                  received_pkt_inner_sqn, phase_data)
                        
                        if current_complete:
                            print(f"串口{ser_id} - pkt_sqn {received_pkt_sqn} 已收到完整inner_sqn序列")
                            
                            both_complete = check_both_serials_complete(received_pkt_sqn)
                            
                            if both_complete:
                                trigger_function(received_pkt_sqn)
                        
                        rawFrame = []
                    else:
                        rawFrame = []
                        
        except Exception as e:
            print(f"串口{ser_id}读取错误: {e}")
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

# 主线程保持运行
try:
    while shared_data['trigger_count'] < shared_data['max_triggers']:
        time.sleep(0.1)
    print("程序完成！")
except KeyboardInterrupt:
    print("程序被用户中断")