import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii
import threading

import matplotlib.pyplot as plt
from math import pi, atan2, sqrt
from scipy.linalg import eig
import time
from sklearn.linear_model import LinearRegression

ser1 = serial.Serial('COM14', 115200)
ser2 = serial.Serial('COM16', 115200)

SPEED_OF_LIGHT  = 299792458
num_iterations = 50     # 进行的循环次数
iteration = 0


music_list = []
grid_search_list = []
esprit_list = []
    
num_samples = 88


ser1_data = {
    'interval': [-1],
    'tx1_sqn': [-1],
    'tx2_sqn': [-1],
    'packet_1_I_data': [-1],
    'packet_1_Q_data': [-1],
    'packet_2_I_data': [-1],
    'packet_2_Q_data': [-1]
}

ser2_data = {
    'interval': [-1],
    'tx1_sqn': [-1],
    'tx2_sqn': [-1],
    'packet_1_I_data': [-1],
    'packet_1_Q_data': [-1],
    'packet_2_I_data': [-1],
    'packet_2_Q_data': [-1]
}
def thread(ser, id):
    rawFrame = []
    num_samples = 88
    while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 4*num_samples*2+9:
                received_data_1 = rawFrame[:4*num_samples]
                received_data_2 = rawFrame[4*num_samples:4*num_samples*2]
                num_samples = 88
                
                interval = struct.unpack('>HH', bytes(rawFrame[4*num_samples*2:-5]))[1]
                tx1_sqn = rawFrame[-5]
                tx2_sqn = rawFrame[-4]

                packet_1_I_data = np.zeros(num_samples, dtype=np.int16)
                packet_1_Q_data = np.zeros(num_samples, dtype=np.int16)
                packet_2_I_data = np.zeros(num_samples, dtype=np.int16)
                packet_2_Q_data = np.zeros(num_samples, dtype=np.int16)

                for i in range(num_samples):
                    (packet_1_I) = struct.unpack('>h', bytes(received_data_1[4*i+2:4*i+4]))
                    (packet_1_Q) = struct.unpack('>h', bytes(received_data_1[4*i:4*i+2]))

                    (packet_2_I) = struct.unpack('>h', bytes(received_data_2[4*i+2:4*i+4]))
                    (packet_2_Q) = struct.unpack('>h', bytes(received_data_2[4*i:4*i+2]))
                    #print(phase)
                    #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                    #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                    packet_1_I_data[i] = packet_1_I[0]
                    packet_1_Q_data[i] = packet_1_Q[0]

                    packet_2_I_data[i] = packet_2_I[0]
                    packet_2_Q_data[i] = packet_2_Q[0]

                packet_1_I_data = packet_1_I_data.astype(np.float32)
                packet_1_Q_data = packet_1_Q_data.astype(np.float32)

                packet_2_I_data = packet_2_I_data.astype(np.float32)
                packet_2_Q_data = packet_2_Q_data.astype(np.float32)

                #print('phase_{}_1:'.format(id), np.arctan2(packet_1_I_data[1], packet_1_Q_data[1]))

                #all_data['I_data'] = I_data
                #all_data['Q_data'] = Q_data
                if id == 1:
                    ser1_data['interval'][0] = interval
                    ser1_data['tx1_sqn'][0] = tx1_sqn
                    ser1_data['tx2_sqn'][0] = tx2_sqn
                    ser1_data['packet_1_I_data'][0] = packet_1_I_data
                    ser1_data['packet_1_Q_data'][0] = packet_1_Q_data
                    ser1_data['packet_2_I_data'][0] = packet_2_I_data
                    ser1_data['packet_2_Q_data'][0] = packet_2_Q_data
                    break
                else:
                    ser2_data['interval'][0] = interval
                    ser2_data['tx1_sqn'][0] = tx1_sqn
                    ser2_data['tx2_sqn'][0] = tx2_sqn
                    ser2_data['packet_1_I_data'][0] = packet_1_I_data
                    ser2_data['packet_1_Q_data'][0] = packet_1_Q_data
                    ser2_data['packet_2_I_data'][0] = packet_2_I_data
                    ser2_data['packet_2_Q_data'][0] = packet_2_Q_data
                    break
                #np.savez('data_{}.npz'.format(id), **all_data)
            rawFrame = []
    #np.savez('100_data_{}.npz'.format(id), **all_data)
    # np.savez('10_data_10_degree{}.npz'.format(id), **all_data)
        

def check_data_completeness_and_validity(ser1_data, ser2_data):
    print('ser1_tx1_sqn:', ser1_data['tx1_sqn'])
    print('ser1_tx2_sqn:', ser1_data['tx2_sqn'])
    print('ser2_tx1_sqn:', ser2_data['tx1_sqn'])
    print('ser2_tx2_sqn:', ser2_data['tx2_sqn'])
    if ser1_data['tx1_sqn'] == ser2_data['tx1_sqn'] and ser1_data['tx2_sqn'] == ser2_data['tx2_sqn']:
        if ser1_data != -1 and ser2_data != -1:
            return True
        else:
            return False
    else:
        return False

def cal_slope(phase_diff):
    x = np.arange(len(phase_diff)).reshape(-1, 1)
    y = phase_diff
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_


def data_process(ser1_data, ser2_data):
    rx1_packet1_I = ser1_data['packet_1_I_data'][0]
    rx1_packet1_Q = ser1_data['packet_1_Q_data'][0]

    rx1_packet2_I = ser1_data['packet_2_I_data'][0]
    rx1_packet2_Q = ser1_data['packet_2_Q_data'][0]

    interval1 = ser1_data['interval'][0]

    rx2_packet1_I = ser2_data['packet_1_I_data'][0]
    rx2_packet1_Q = ser2_data['packet_1_Q_data'][0]

    rx2_packet2_I = ser2_data['packet_2_I_data'][0]
    rx2_packet2_Q = ser2_data['packet_2_Q_data'][0]

    interval2 = ser2_data['interval'][0]

    interval = (interval1+interval2)/2

    rx1_pkt1_phase = np.unwrap(np.arctan2(rx1_packet1_Q, rx1_packet1_I))
    rx1_pkt2_phase = np.unwrap(np.arctan2(rx1_packet2_Q, rx1_packet2_I))
    rx2_pkt1_phase = np.unwrap(np.arctan2(rx2_packet1_Q, rx2_packet1_I))
    rx2_pkt2_phase = np.unwrap(np.arctan2(rx2_packet2_Q, rx2_packet2_I))

    packet1_phase_diff = rx1_pkt1_phase - rx2_pkt1_phase
    packet2_phase_diff = rx1_pkt2_phase - rx2_pkt2_phase

    packet1_phase_diff_unwrap = np.unwrap(packet1_phase_diff)
    packet2_phase_diff_unwrap = np.unwrap(packet2_phase_diff)

    packet1_slope, packet1_intercept = cal_slope(packet1_phase_diff_unwrap)
    packet2_slope, packet2_intercept = cal_slope(packet2_phase_diff_unwrap)


    pkt1_geo_diff = 0
    drift_t1 = packet1_intercept - pkt1_geo_diff
    drift_delta_t = drift_t1 + interval/16*packet1_slope

    pkt2_geo_diff = packet2_intercept - drift_delta_t
    pkt2_geo_diff = np.arctan2(np.sin(pkt2_geo_diff), np.cos(pkt2_geo_diff))
    
    if pkt2_geo_diff/(2*np.pi)*12.5/6 > 1:
        angle = 90
    elif pkt2_geo_diff/(2*np.pi)*12.5/6 < -1:
        angle = -90
    else:
        angle = np.arcsin(pkt2_geo_diff/(2*np.pi)*12.5/6)/np.pi*180
    
    print(angle)
    return  angle

def start_monitoring(ser1, ser2):
    global ser1_data, ser2_data
    print("串口监控已启动...")
    
    angle_list = []

    while len(angle_list)<=50:
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
            angle = data_process(ser1_data, ser2_data)
            angle_list.append(angle)
        else:
            # 数据无效，清除数据重新开始
            print("数据无效，重新开始监控...")
            ser1_data = {
                'interval': [-1],
                'tx1_sqn': [-1],
                'tx2_sqn': [-1],
                'packet_1_I_data': [-1],
                'packet_1_Q_data': [-1],
                'packet_2_I_data': [-1],
                'packet_2_Q_data': [-1]
            }

            ser2_data = {
                'interval': [-1],
                'tx1_sqn': [-1],
                'tx2_sqn': [-1],
                'packet_1_I_data': [-1],
                'packet_1_Q_data': [-1],
                'packet_2_I_data': [-1],
                'packet_2_Q_data': [-1]
            }
        
        # 短暂延迟后继续下一轮
        time.sleep(0.1)
        print(f"当前已触发次数{len(angle_list)}")
    print(f"达到最大触发次数{len(angle_list)},监控结束")
    np.savez('experiment_data/angle_{}.npz'.format(-45), angle_list)



# 启动监控
start_monitoring(ser1, ser2)