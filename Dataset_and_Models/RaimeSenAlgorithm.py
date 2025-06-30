# -*- coding: utf-8 -*-
"""
Editor de Spyder

@author: Daivd093

Este archivo está basado en la implementación de raimasen1729
https://github.com/raimasen1729/Finger-Flexion-Detection-Using-ECoG-Signal/tree/main
para detectar la flexión de un dedo usando señales ECoG.

La versión actual solo considera al sujeto 1 y genera 5 redes neuronales independientes
para predecir la posición de cada uno de sus dedos.

Se hizo usando  Python 3.9.18
                Scipy 1.13.1
                Numpy 1.26.4
                MatPlotLib 3.9.2
                Tensorflow 2.10.0
                scikit-learn 1.6.1

                
Estoy usando subject2 como si fuese subject1, luego lo arreglaré
"""
import time
START_TIME = time.time()
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



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


logfile = logdir+'entrenamiento_ecog_info.log'
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
from scipy.interpolate import CubicSpline
from scipy.io import savemat
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
import matplotlib.pyplot as plt
logging.info(f'Numpy {np.__version__}')
logging.info(f'MatPlotLib {plt.matplotlib.__version__}')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
logging.info(f'Tensorflow {tf.__version__}')

import joblib

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler #standard scaler worked best
logging.info(f'scikit-learn {sklearn.__version__}')



#Preprocess:
from utils import  get_windowed_feats, get_windowed_dg, create_R_matrix
    

# Train / Test split

#----------SUBJECT--------------
s = int(input("¿Para qué sujeto del dataset 4 de la competencia BCI IV quiere realizar el entrenamiento? [1/2/3] ")) # Me dio flojera hacer algo más robusto ahora

filename = f"Dataset4_BCICIV/sub{s}_processed.npz"
logging.info(f"Esta versión va a intentar cargar {filename}, si no lo logra, va a hacer el preprocesamiento del dataset.")


test_data_shape_0 = 200000 # Debería haber incluído este dato en el .npz, por ahora va hardcodeado no más


try:
    with np.load(filename) as data:
        R_train = data["R_train"]
        R_test=data['R_test']
        y_train=data["y_train"]
        y_test=data["y_test"]
        test_data_shape_0 = data["test_data_shape_0"]
    logging.info(f"Dataset preprocesado cargado desde {filename} con éxito!")
    print("Si se pudo")
except:
    logging.info("No se pudo cargar dataset preprocesado")
    print("No se pudo")
    
    subject = loadmat(f'Dataset4_BCICIV/sub{str(s)}_comp.mat')
    
    train_dg = subject['train_dg']
    train_data = subject['train_data']
    test_data = subject['test_data']
    
    #Preprocess: Removing bad channels
    
    if s == 1:
        # We have to remove chanel 54 from subject1
        train_data = np.delete(train_data, 54,1)
        test_data = np.delete(test_data, 54,1)
    elif s == 2:
        # We have to remove chanels 20 and 37 from subject2
        train_data = np.delete(train_data, [20,37],1)
        test_data = np.delete(test_data, [20,37],1)
    elif s == 3:
        # We don't need to remove any channels from subject3
        pass
    else:
        logging.error("El sujeto no existe")
        # Probablemente debería poner algo para detener el programa sin que tire un error
    
    test_data_shape_0 = test_data.shape[0]
    
    fs = 1000 #Hz
    
    #X_train
    X_train = get_windowed_feats(train_data, fs, 0.2,0.05)
    X_train = np.nan_to_num(X_train)
    logging.info('X_train Listo!')
    logging.debug(f"X_train shape: {X_train.shape}")
    
    
    #X_test
    X_test = get_windowed_feats(test_data, fs, 0.2,0.05)
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
    

# Neural Network

def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim = R_train.shape[1], activation = 'relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation = 'relu', kernel_regularizer=l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))

    #choosing optimizer and performance metric
    model.compile(optimizer = 'adam', loss = Huber(delta=0.5))
    
    return model


# early stopping
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose=1, patience = 25)
# reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

formatted = lambda et : time.strftime("%H:%M:%S", time.gmtime(et))

#----------SUBJECT-------------

logging.info('Se aplica StandardScaler a R usando R_train')
scaler = StandardScaler()
R_train = scaler.fit_transform(R_train)

scaler_filename = f"scaler{str(s)}.save"
joblib.dump(scaler, scaler_filename) 

R_test = scaler.transform(R_test)
logging.info('Se aplica MinMaxScaler a y_train')

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)


logging.info("Se hace split temporal dividiendo 90/10 train/val")
# Split temporal antes de entrenar 
split_idx = int(0.9 * len(R_train))
R_train_part = R_train[:split_idx]
y_train_part = y_train[:split_idx, :]


R_val_part = R_train[split_idx:]
y_val_part = y_train[split_idx:, :]



logging.debug(f"R_train_part shape: {R_train_part.shape}")
logging.debug(f"R_val_part shape: {R_val_part.shape}")

logging.debug(f"y_train_part shape: {y_train_part.shape}")
logging.debug(f"y_val_part shape: {y_val_part.shape}")




test_preds = []
for finger in range(5):

    start = time.time()
    model = build_model()
    model.fit(R_train_part, y_train_part[:,finger],
                    validation_data=(R_val_part, y_val_part[:,finger]),
                    epochs=250, batch_size=32,
                    callbacks=[early_stop, reduce_lr], verbose=1)
    end = time.time()
    model.save(f'models/model_s{s}f{str(finger+1)}.h5')
    #predictions
    test_preds.append(model.predict(R_test))
    
    et = -start+end
    logging.info(f"Duración de entrenamiento para el dedo {finger+1} del sujeto {s}: {formatted(et)}")


# Predictions

test_pred_s_scaled = np.hstack(test_preds)
logging.debug(f"test_pred_s_scaled shape: {test_pred_s_scaled.shape}")
test_pred_s = scaler_y.inverse_transform(test_pred_s_scaled)

xs = np.linspace(0,test_pred_s.shape[0],test_data_shape_0)

#interpolation
y = np.empty_like(test_pred_s[:,0]) 
cs = [] 
for i in range(test_pred_s.shape[1]):
    x = np.arange(test_pred_s.shape[0])
    y = test_pred_s[:,i]
    cs.append(CubicSpline(x, y, bc_type = 'clamped'))

interp_pred_nn = np.vstack((cs[0](xs),cs[1](xs), cs[2](xs), cs[3](xs), cs[4](xs))).T

folder = 'Predictions'
try:
    os.mkdir(folder)
    logging.info(f"Carpeta {folder} Creada")
except:
    logging.info(f"Carpeta {folder} ya existía")
    
filename = folder + f'/subj{s}_testpredictions.mat'
mat_dict = {'predicted_dg': interp_pred_nn}
savemat(filename, mat_dict)

logging.info(f"Predicciones guardadas en {filename}")


END_TIME = time.time()

TOTAL_TIME = END_TIME-START_TIME

logging.info(f"Duración total del programa: {formatted(TOTAL_TIME)}")