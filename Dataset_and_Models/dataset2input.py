"""
@author: Daivd093

Este archivo está basado en la implementación de raimasen1729
https://github.com/raimasen1729/Finger-Flexion-Detection-Using-ECoG-Signal/tree/main
para detectar la flexión de un dedo usando señales ECoG.

Este archivo solo se encarga del preprocesamiento de los datos y hace los cálculos de
la matriz R. Esencialmente, recibe el dataset y entrega la matriz que funciona como entrada
para la red neuronal.


Se hizo usando  Python 3.9.18
                Scipy 1.13.1
                Numpy 1.26.4
                MatPlotLib 3.9.2
                Tensorflow 2.10.0
                scikit-learn 1.6.1

     

Esta versión convierte a float32 antes de guardar, más adelante podría guardar 
ambas o preguntar. Debo agregar en el gitignore que no suba los arreglos de numpy
           
"""

import time
tini = time.time()


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


logfile = logdir+'dataset2input_ecog_info.log'
logging.basicConfig(
    filename=logfile,     # nombre del archivo de log
    level=logging.INFO,                    # nivel mínimo a registrar
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # o 'a' para agregar sin sobrescribir
)

logging.info("======================Inicio======================")

print(logfile)
import sys
logging.info(f'Python {sys.version.split()[0]}')

import os

import scipy
from scipy.io import loadmat
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
import matplotlib.pyplot as plt
logging.info(f'Numpy {np.__version__}')
logging.info(f'MatPlotLib {plt.matplotlib.__version__}')

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler #standard scaler worked best
logging.info(f'scikit-learn {sklearn.__version__}')

import joblib

from utils import get_windowed_feats, get_windowed_dg, create_R_matrix

dataset_dir = "Dataset4_BCICIV/" # Luego debería estandarizar esto para que todo se corra desde root o todo desde la carpeta en la que está
subject = input("¿Para qué sujeto del dataset 4 de la competencia BCI IV quiere hacer la conversión? [1/2/3]? ")
assert subject in ["1","2","3"]

dataset_file = f"{dataset_dir}sub{subject}_comp.mat"
testlabels_file = f"{dataset_dir}sub{subject}_testlabels.mat"
logging.info(f"Se hará la conversión para {dataset_file}")


dataset = loadmat(dataset_file)

test_data = dataset ['test_data']
train_data = dataset ['train_data']
train_dg = dataset ['train_dg']

testlabels = loadmat(testlabels_file)
test_dg = testlabels['test_dg']

Remove = input("¿Qué canales deseas eliminar? [54/20 37/]\n(Si quiere más se uno, separe los números con un espacio y si no quiere ninguno, no ingrese nada) ")
if Remove != '':
    Remove = Remove.split(" ")
    Remove = [int(cr) for cr in Remove]
else:
    Remove = []
logging.info(f"Se eliminarán los canales {Remove}")



#X_train
X_train = get_windowed_feats(train_data, 1000, 0.2,0.05)
X_train = np.nan_to_num(X_train)
logging.info('X_train Listo!')
logging.debug(f"X_train shape: {X_train.shape}")


#X_test
X_test = get_windowed_feats(test_data, 1000, 0.2,0.05)
X_test = np.nan_to_num(X_test)
logging.info('X_test Listo!')
logging.debug(f"X_test shape: {X_test.shape}")

#y_train
y_train = get_windowed_dg(train_dg, 25, 7, 2)
logging.info('y_train Listo!')
logging.debug(f"y_train shape: {y_train.shape}")


# APLICAR LAG
N_wind = 3
LAG = 1
X_train = X_train[:-LAG, :]
y_train = y_train[LAG:, :]

#R calc
R_train = create_R_matrix(X_train, 3)
logging.info('R_train Listo!')
logging.debug(f"R_train shape: {R_train.shape}")
R_test = create_R_matrix(X_test, 3)
logging.info('R_test Listo!')
logging.debug(f"R_test shape: {R_test.shape}")


#y_test 
y_test = get_windowed_dg(test_dg, 25, 7, 2)
logging.info('y_test Listo!')
logging.debug(f"y_test shape: {y_test.shape}")



logging.info('Se aplica StandardScaler a R usando R_train')
scaler_R = StandardScaler()
R_train = scaler_R.fit_transform(R_train)
R_test = scaler_R.transform(R_test)

scaler_R_filename = f"scaler_R_{subject}.save"
joblib.dump(scaler_R, scaler_R_filename) 
logging.info(f"Se guarda el StandardScaler de R en {scaler_R_filename}")



logging.info('Se aplica MinMaxScaler a y_train')
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)

scaler_y_filename = f"scaler_y_{subject}.save"
joblib.dump(scaler_y, scaler_y_filename) 
logging.info(f"Se guarda el MinMaxScaler de y en {scaler_y_filename}")



logging.warning("Esta versión convierte a float32 antes de guardar")

y_train = y_train.astype(np.float32)
logging.debug("y_train convertido a float32")

R_train = R_train.astype(np.float32)
logging.debug("R_train convertido a float32")

R_test = R_test.astype(np.float32)
logging.debug("R_test convertido a float32")

y_test = y_test.astype(np.float32)
logging.debug("y_test convertido a float32")


filename = f"{dataset_dir}sub{subject}_processed.npz"
np.savez(filename,R_train=R_train,R_test=R_test,y_train=y_train,y_test=y_test,test_data_shape_0=test_data.shape[0])
logging.info(f"Guardados en {filename}")



tfin = time.time()

formatted = lambda et : time.strftime("%H:%M:%S", time.gmtime(et))

logging.debug(f"Tiempo total transcurrido: {formatted(tfin-tini)}")