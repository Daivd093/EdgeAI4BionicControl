import serial
import time

puerto = 'COM3'
baudrate = 115200
ser = serial.Serial(puerto, baudrate, timeout=5)
time.sleep(5)

x = []
y = []

for i in range(9):    
    valor = 0.01*(i+1)
    print(valor)
    ser.write(f"{valor:.2f}".encode())

    respuesta = ser.readline().decode().strip()
    if not respuesta:
        print("Nada")
        continue
    print("STM32 respondió:", respuesta)

    xy = respuesta.strip(" ()").split(',')
    x.append(float(xy[0]))
    y.append(float(xy[1]))
    

print("x = ", x)
print("y = ", y)
ser.close()
