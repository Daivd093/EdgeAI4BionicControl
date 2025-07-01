# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 22:03:49 2025

@author: david

Este archivo carga las matrices R y envía fila por fila via UART
"""


import os
logdir = 'logs/'
try:
    os.mkdir(logdir)
except FileExistsError:
    pass  # ya existe, no pasa nada

import logging


# Limpiar handlers anteriores
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logfile = logdir+'envio_uart.log'
logging.basicConfig(
    filename=logfile,     # nombre del archivo de log
    level=logging.INFO,                    # nivel mínimo a registrar
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',  # o 'a' para agregar sin sobrescribir
    encoding='utf-8' # Para los tildes
)

logging.info("======================Inicio======================")

print(logfile)
import sys
logging.info(f'Python {sys.version.split()[0]}')

import os


import scipy
from scipy.io import savemat
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
logging.info(f'Numpy {np.__version__}')


s = input("¿Los datos de qué sujeto desea enviar por UART? [1/2/3]: ")

filename = f"Dataset4_BCICIV/sub{s}_processed.npz"

try:
    with np.load(filename) as data:
#        R_train = data["R_train"]
        R_test=data['R_test']
#        y_train=data["y_train"]
        y_test=data["y_test"]
#        test_data_shape_0 = data["test_data_shape_0"]
    logging.info(f"Dataset preprocesado cargado desde {filename} con éxito!")
    logging.info("Esta versión solo envía las filas de R_test.")

except:
    logging.error(f"No se ha hayado el archivo {filename} con el dataset preprocesado.\nEs necesario ejecutar dataset2input.py si este no ha sido generado.")
    assert False

N = R_test.shape[1] 
logging.info(f"Se enviarán {N} {type(R_test[0,0])}")
N = N*4    
    
L = R_test.shape[0]

pred_stm32 = []

import serial

puerto = 'COM3'
baudrate = 115200
ser = serial.Serial(puerto, baudrate, timeout=1)



logging.warning("Esta cosa está funcionando a penas y luego será corregido")

# Saludo
respuesta = ser.readline().decode()
while respuesta:
    #print(respuesta)
    respuesta = ser.readline().decode()
    
# Envía tamaño
ser.write(f"{N:04}".encode())

# Espera recibir
respuesta = ser.readline().decode()
while respuesta:
    #print(respuesta)
    respuesta = ser.readline().decode()


for fila in range(L):
    
    # Envía una fila
    ser.write(R_test[fila].tobytes())
    
    # Respuesta
    respuesta = ser.readline().decode()
    try:
        pred_stm32.append(np.float32(respuesta))
    except:
        pass

    while respuesta:
        #print(respuesta)
        respuesta = ser.readline().decode()
    

ser.close()



folder = 'Predictions'
try:
    os.mkdir(folder)
    logging.info(f"Carpeta {folder} Creada")
except:
    logging.info(f"Carpeta {folder} ya existía")
    
filename = folder + f'/subj{s}_stm32_testpredictions.mat'
mat_dict = {'unprocessed_predictions': pred_stm32}
savemat(filename, mat_dict)