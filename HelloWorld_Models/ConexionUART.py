import serial
import time

import numpy as np
import matplotlib.pyplot as plt
    

V = int(input("¿Cuántas veces quieres correr esto?\n"))

X = []
Y = []
L = []

for _ in range(V):
    S = input("¿Sigo? [y]/n\n")
    if (S.lower()=="n"):
        break
    
    l = input("¿Qué leyenda quiere ponerle a este modelo?\n")
    L.append(l)

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


    plt.plot(x,y)
    plt.show()

    X.append(x)
    Y.append(y)




Ref = input("¿Quiere agregar referencia sin(x)? [y]/n")
if (Ref.lower() != "n"):
    N = 10000
    x = np.linspace(0,2*np.pi,N)
    y = np.sin(x)

    X.append(x)
    Y.append(y)

    V+=1
    L.append("sin(x)")


legends = []
for k in range(V):
    plt.plot(X[k],Y[k])
    legends.append(L[k])
    
plt.legend(legends)
plt.show()