import serial
import time
import numpy as np

puerto = 'COM3'
baudrate = 115200
ser = serial.Serial(puerto, baudrate, timeout=5)
time.sleep(5)

x = []
y = []

N = 10000
for i in np.linspace(0,2*np.pi,N):    
    
    ser.write(f"{i:.5f}".encode())

    respuesta = ser.readline().decode().strip()
    if not respuesta:
        print("Nada")
        continue
    #print("STM32 respondió:", respuesta)

    xy = respuesta.strip(" ()").split(',')
    x.append(float(xy[0]))
    y.append(float(xy[1]))
    

#print("x = ", x)
#print("y = ", y)
ser.close()

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()