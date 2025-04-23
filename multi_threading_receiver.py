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

def thread(ser, id):
    
    all_data = {
        'packet_1_I_data': [],
        'packet_1_Q_data': [],
        'packet_2_I_data': [],
        'packet_2_Q_data': [],
        'interval': []
    }
    rawFrame = []
    num_samples = 88
    cte_number = 0
    while cte_number < 100:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 4*num_samples*2+7:
                received_data_1 = rawFrame[:4*num_samples]
                received_data_2 = rawFrame[4*num_samples:4*num_samples*2]
                num_samples = 88
                
                interval = struct.unpack('>hh', bytes(rawFrame[4*num_samples*2:-3]))[1]

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
                all_data['packet_1_I_data'].append(packet_1_I_data)
                all_data['packet_1_Q_data'].append(packet_1_Q_data)
                all_data['packet_2_I_data'].append(packet_2_I_data)
                all_data['packet_2_Q_data'].append(packet_2_Q_data)
                all_data['interval'].append(interval)
                cte_number += 1
                print('{}'.format(id), cte_number)
                #np.savez('data_{}.npz'.format(id), **all_data)
                rawFrame = []
    #np.savez('100_data_{}.npz'.format(id), **all_data)
    np.savez('100_data_10_degree{}.npz'.format(id), **all_data)
        
            

thread1 = threading.Thread(target=thread, args=(ser1,1))
thread2 = threading.Thread(target=thread, args=(ser2,2))
thread1.start()
thread2.start()