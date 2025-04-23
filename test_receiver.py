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

ser = serial.Serial('COM16', 115200)

SPEED_OF_LIGHT  = 299792458
num_iterations = 50     # 进行的循环次数
iteration = 0

rawFrame = []

all_data = {
    'I_data': [],
    'Q_data': []
}

music_list = []
grid_search_list = []
esprit_list = []
    


phased0_list = []
    
rawFrame = []
while len(phased0_list) <= 1000:
    byte  = ser.read(1)        
    rawFrame += byte
    print(rawFrame)
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 7:
            received_data = rawFrame[:4]
            I_data = np.zeros(1, dtype=np.int16)
            Q_data = np.zeros(1, dtype=np.int16)
            for i in range(1):
                (I) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                (Q) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))
                #print(phase)
                #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                I_data[i] = I[0]
                Q_data[i] = Q[0]

            I_data = I_data.astype(np.float32)
            Q_data = Q_data.astype(np.float32)

            print('phase_0:', np.arctan2(Q_data[0], I_data[0]))
            phased0_list.append(np.arctan2(Q_data[0], I_data[0]))
            #print('phase_{}_1:'.format(id), np.arctan2(I_data[1], Q_data[1]))
            
            #print('I_data_{}:'.format(id), I_data[0])
            storage_I_data = I_data
            storage_Q_data = Q_data

            all_data['I_data'] = I_data
            all_data['Q_data'] = Q_data

            #np.savez('data_{}.npz'.format(id), **all_data)
            #plt.plot(64*np.arctan2(Q_data, I_data), marker='.')
            #plt.show()


        rawFrame = []


plt.boxplot(phased0_list)
plt.show()
