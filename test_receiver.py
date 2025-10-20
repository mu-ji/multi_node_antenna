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

ser1 = serial.Serial('COM14', 115200)
ser2 = serial.Serial('COM16', 115200)

SPEED_OF_LIGHT  = 299792458
num_iterations = 50     # 进行的循环次数
iteration = 0


music_list = []
grid_search_list = []
esprit_list = []
    
num_samples = 88

rx1_pkt_list = []
rx2_pkt_list = []

frame_num = 2


def thread(ser, id):
    
    rawFrame = []
    num_samples = 88
    cte_number = 0
    
    # 为每个线程创建独立的数据列表
    if ser == ser1:
        rx_pkt_list = []  # 线程1的独立列表
    elif ser == ser2:
        rx_pkt_list = []  # 线程2的独立列表
    
    while cte_number < 6*frame_num:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-4:]==[255, 255, 255, 255]:
            if len(rawFrame) == 4*num_samples+10:
                received_data = rawFrame[:4*num_samples]
                received_timestamp = rawFrame[4*num_samples:4*num_samples+4]
                received_pkt_sqn = rawFrame[4*num_samples+4]
                received_pkt_inner_sqn = rawFrame[4*num_samples+5]

                received_timestamp = struct.unpack('>I', bytes(received_timestamp))[0]
                print(received_timestamp)
                print(received_pkt_sqn)
                print(received_pkt_inner_sqn)

                # extract IQ data
                packet_I_data = np.zeros(num_samples, dtype=np.int16)
                packet_Q_data = np.zeros(num_samples, dtype=np.int16)
                for i in range(num_samples):
                    (packet_I) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                    (packet_Q) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))

                    packet_I_data[i] = packet_I[0]
                    packet_Q_data[i] = packet_Q[0]

                packet_I_data = packet_I_data.astype(np.float32)
                packet_Q_data = packet_Q_data.astype(np.float32)

                # 关键：每次循环都创建新的字典对象
                pkt_info = {
                    'I_data': packet_I_data.copy(),      # 使用copy()确保数据独立
                    'Q_data': packet_Q_data.copy(),      # 使用copy()确保数据独立
                    'timestamp': received_timestamp,
                    'pkt_sqn': received_pkt_sqn,
                    'pkt_inner_sqn': received_pkt_inner_sqn
                }

                # 添加到当前线程的列表
                rx_pkt_list.append(pkt_info)

                cte_number += 1
                rawFrame = []
    
    # 保存数据
    if ser == ser1:
        rx1_pkt_array = np.array(rx_pkt_list)
        np.savez('data/rx1_data_{}.npz'.format(frame_num), rx1_pkt_array)
    elif ser == ser2:
        rx2_pkt_array = np.array(rx_pkt_list)
        np.savez('data/rx2_data_{}.npz'.format(frame_num), rx2_pkt_array)
        
            

thread1 = threading.Thread(target=thread, args=(ser1,1))
thread2 = threading.Thread(target=thread, args=(ser2,2))
thread1.start()
thread2.start()


