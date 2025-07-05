"""
@author: DaivdTP

Este archivo contiene código para probar múltiples modelos neuronales via UART.
Este archivo se ejecuta en el PC y se define cuántas veces se quiere correr (cuántas redes neuronales se quieren evaluar)
Luego, asegurando que haya una conexión por UART con una red neuronal que entrega sus respuestas del modo (input,output), 
se deja que el programa envíe 10000 números entre 0 y 2 pi. Al terminar muestra una gráfica de los resultados y pide confirmación
antes de volver a ejecutarse. 
Se debe cambiar la red neuronal del microcontrolador antes de confirmar que se quiere seguir.

Al terminar se pregunta si se quiere añadir la función verdadera (en este caso, el seno de x) para luego graficar todos los modelos y la referencia.

Finalmente, se calcula el mse asociado a cada modelo. La versión actual calcula el mse con respecto al seno de x y no revisa si la última entrada
es la referencia o no.

Versiones posteriores de esto podrían aumentar robustez
Podrían incluir alguna especie de pantalla de carga
Podrían revisar si el último modelo es la referencia
"""


import serial
import time

import numpy as np
import matplotlib.pyplot as plt
    
import pickle

V = int(input("¿Cuántas veces quieres correr esto? "))

X = []
Y = []
L = []

for _ in range(V):
    S = input("¿Sigo? [y]/n\n")
    if (S.lower()=="n"):
        break
    
    l = input("¿Qué leyenda quiere ponerle a este modelo? ")
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




Ref = input("¿Quiere agregar referencia sin(x)? [y]/n ")
if (Ref.lower() != "n"):
    N = 10000
    x = np.linspace(0,2*np.pi,N)
    y = np.sin(x)

    X.append(x)
    Y.append(y)

    V+=1
    L.append("sin(x)")

Save = input("¿Quiere guardar X,Y,L? [y]/n ")
if (Ref.lower() != "n"):
    with open("datos.pkl", "wb") as f:
        pickle.dump((X, Y, L), f)


# Para leer es simplemente
Load = input('¿Quiere cargar los datos.pkl? y/[n] ')
if (Load.lower() == "y"):

    with open("datos.pkl", "rb") as f:
        X, Y, L = pickle.load(f)
    
    assert len(L) == len(X)
    assert len(Y) == len(X)
    V = len(L)



legends = []
for k in range(V):
    plt.plot(X[k],Y[k])
    legends.append(L[k])
    
plt.legend(legends)
plt.show()


# Para ver el error cuadrático medio, considerando que la función real es el seno de X:
# (Esto podría perfectamente ir en un archivo aparte)
X = np.array(X)
Y = np.array(Y)

print(X[0].shape)

for k in range(V):
    assert len(X[k]) == len(Y[k])

    N = len(X[k])

    mae_model = 1/N * (sum(abs(Y[k]-np.sin(X[k]))))

    print(f"MAE_{L[k]} = {mae_model}\n")