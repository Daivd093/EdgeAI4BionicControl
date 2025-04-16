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
from scipy import signal as sig
from scipy.signal import filtfilt
from scipy.signal import firwin
from scipy.signal import kaiserord
from copy import deepcopy
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import CubicSpline
from scipy.io import savemat
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
import matplotlib.pyplot as plt
logging.info(f'Numpy {np.__version__}')
logging.info(f'MatPlotLib {plt.matplotlib.__version__}')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
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


fs = 1000 #Hz

subject1 = loadmat('Dataset4 BCICIV/sub1_comp.mat')
subject2 = loadmat('Dataset4 BCICIV/sub2_comp.mat')
subject3 = loadmat('Dataset4 BCICIV/sub3_comp.mat')

train_dg1 = subject1['train_dg']
train_dg2 = subject2['train_dg']
train_dg3 = subject3['train_dg']

train_data1 = subject1['train_data']
train_data2 = subject2['train_data']
train_data3 = subject3['train_data']

test_data1 = subject1['test_data']
test_data2 = subject2['test_data']
test_data3 = subject3['test_data']


#Preprocess: Removing bad channels

# We have to remove chanel 54 from subject1
train_data1 = np.delete(train_data1, 54,1)
test_data1 = np.delete(test_data1, 54,1)

# We have to remove chanels 20 and 37 from subject2
train_data2 = np.delete(train_data2, [20,37],1)
test_data2 = np.delete(test_data2, [20,37],1)

# We don't need to remove any channels from subject3


#Preprocess: Filtering 

def filter_data_2(raw_eeg, fs=1000):
    nyq=fs/2
    ripple_db=60.0 #ripple size for cutoff
    width=5.0/nyq #transition wid
    N, beta = kaiserord(ripple_db, width) #kaiser parameters
    taps_default=firwin(N,[4, 200],window=('kaiser',beta),pass_zero='bandpass',fs=1000)
    filtered_2=filtfilt(taps_default,1.0,raw_eeg)
    return filtered_2

def filternotch(x,fs=1000):
    b1, a1 = sig.iirnotch(60, 20, fs)
    b2, a2 = sig.iirnotch(120, 20, fs)
    b3, a3 = sig.iirnotch(180, 20, fs)
    output1=sig.filtfilt(b1,a1,x)
    output2=sig.filtfilt(b2,a2,output1)
    output3=sig.filtfilt(b3,a3,output2)
    return output3

#Preprocess: Windowing

def NumWins(x,fs,winLen,winDisp):
    total_time = len(x)/fs
    M =np.floor((total_time - winLen)/ winDisp)
    return int(M)


#Preprocess: Feature Definitions
# Feature 1
def Avg_voltage(x):
    avg_vol = np.mean(x)
    return avg_vol

# Feature 2
def LL(x):
    ll_x = np.sum(np.absolute(np.ediff1d(x)))
    return ll_x

# Feature 3
def Energy(x):
    E_x = np.sum(np.square(x))
    return E_x

# Feature 4 ,5, 6, 7, 8
def Avg_Freq(x, fi, ff):
    #Convert to frequency domain:
    freq_sig = rfft(x)
    N = len(x)
    tdomain=rfftfreq(N,1/fs)  
    indices = np.where((tdomain>=fi) & (tdomain<=ff))[0]
    if len(indices) == 0:
        logging.warning(f"Sin datos en rango {fi}-{ff} Hz")
        return 0.0  # o np.nan
  
    return np.mean(np.abs(freq_sig[indices]))

#Preprocess: Feature Extraction
def get_features(filtered_window, fs=1000):
    channels = np.shape(filtered_window)[1]
    features = np.empty([channels, 8])
    for ch in range(channels):
        feat1 = Avg_voltage(filtered_window[:,ch])
        feat2 = LL(filtered_window[:,ch])
        feat3 = Energy(filtered_window[:,ch])
        feat4 = Avg_Freq(filtered_window[:,ch], 5, 15)
        feat5 = Avg_Freq(filtered_window[:,ch], 20, 25)
        feat6 = Avg_Freq(filtered_window[:,ch], 75, 115)
        feat7 = Avg_Freq(filtered_window[:,ch], 125, 160)
        feat8 = Avg_Freq(filtered_window[:,ch], 160, 175)
        features[ch,:] = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8]
    
    features = np.reshape(features,(channels*8))

    return features

# Preprocess: Windowing Features
def get_windowed_feats(raw_ecog, fs, window_length, window_overlap, clean_nans=True):
    logging.info("Filtrando señal ECoG...")
    filtered_eeg = np.empty(np.shape(raw_ecog))
  
    for ch in range(np.shape(raw_ecog)[1]):
        filtered_eeg[:,ch] = filternotch(filter_data_2(raw_ecog[:,ch]))  
    
    logging.info("Calculando cantidad de ventanas posibles...")
    M = NumWins(filtered_eeg, fs, window_length, window_overlap)
    logging.info(f"Total de ventanas posibles: {M}")
    
    xLen = len(filtered_eeg)
    L = window_length
    d = window_overlap

    feature_vector = []
    for i in range(int(M)):
        start = round(xLen - ((L + i*d) * fs))
        end =round(xLen - (i*d * fs))

        if start >= end or start < 0 or end > xLen:
            logging.warning(f"Ventana {i} fuera de rango (start={start}, end={end}), saltando...")
            continue

        logging.debug(f"Ventana {i}/{int(M)}")
        
        segment = filtered_eeg[start:end, :]
        if segment.shape[0] < fs * 0.2:  # Alerta si es demasiado corto para features como Avg_Freq(20–25)
            logging.warning(f"Ventana {i} demasiado corta: {segment.shape[0]} muestras")

        try:
            feature_values = get_features(segment)
            if np.isnan(feature_values).any():
                logging.warning(f"NaNs detectados en ventana {i}")
            feature_vector.append(feature_values)
        except Exception as e:
            logging.error(f"Falló get_features() en ventana {i} (start={start}, end={end}): {e}")
            continue
        
    feature_vector = np.array(feature_vector)
    feature_vector=feature_vector[::-1,:]


    if clean_nans:
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        logging.info("Reemplazados NaN e Inf por ceros en el feature vector")

    return feature_vector

# Preprocess: Windowing Labels
def get_windowed_dg(raw_dg, fs, window_length, window_overlap):

    M = NumWins(raw_dg, fs, window_length, window_overlap)
    xLen = len(raw_dg)
    L = window_length
    d = window_overlap
    
    downsampled_dg = []

    for i in range(M):
        start = round(xLen - ((L + i*d) * fs))
        end = round(xLen - (i * d * fs))

        if start < 0 or start >= end:
            logging.warning(f"Etiquetas: ventana {i} fuera de rango (start={start}, end={end})")
            continue

        feats = []
        for ch in range(5):
            segment = raw_dg[start:end, ch]
            if segment.size == 0:
                feat = 0.0
            elif np.isnan(segment).any():
                logging.warning(f"NaNs en y_train ventana {i}, canal {ch}, usando np.nanmean")
                feat = np.nanmean(segment)
            else:
                feat = np.mean(segment)
            feats.append(feat)
            
        downsampled_dg.append(feats)

    downsampled_dg = np.array(downsampled_dg)[::-1, :]
    return downsampled_dg


# Preprocess: Regression Matrix
def create_R_matrix(features, N_wind):

    M, ch = np.shape(features)
    feats2 = np.empty([M+N_wind-1, ch])
  
    feats2[N_wind-1:,:] = deepcopy(features)
    for i in range(0,N_wind-1):
        feats2[i, :] = deepcopy(features[i,:])
  
    R = np.empty([M, ch*N_wind + 1])
    R[:,0] = np.ones((M))
  
    for i in range(N_wind-1, np.shape(feats2)[0]):
        temp_arr = []
        for n in np.arange(0,N_wind)[::-1]:
            temp_arr = np.concatenate((temp_arr, feats2[i-(n),:]), axis=None)
    
        R[i-(N_wind-1), 1:] = temp_arr

    return R

# Train / Test split

#----------SUBJECT 1-------------

#X_train
X_train_1 = get_windowed_feats(train_data1, 1000, 0.2,0.05)
X_train_1 = np.nan_to_num(X_train_1)
logging.info('X_train_1 Listo!')
logging.debug(f"X_train_1 shape: {X_train_1.shape}")


#X_test
X_test_1 = get_windowed_feats(test_data1, 1000, 0.2,0.05)
X_test_1 = np.nan_to_num(X_test_1)
logging.info('X_test_1 Listo!')
logging.debug(f"X_test_1 shape: {X_test_1.shape}")

#y_train
y_train_1 = get_windowed_dg(train_dg1, 25, 7, 2)
logging.info('y_train_1 Listo!')
logging.debug(f"y_train_1 shape: {y_train_1.shape}")


# APLICAR LAG
N_wind = 3
LAG = 1
X_train_1 = X_train_1[:-LAG, :]
y_train_1 = y_train_1[LAG:, :]



#R calc
R1_train = create_R_matrix(X_train_1, 3)
logging.info('R1_train Listo!')
logging.debug(f"R1_train shape: {R1_train.shape}")
R1_test = create_R_matrix(X_test_1, 3)
logging.info('R1_test Listo!')
logging.debug(f"R1_test shape: {R1_test.shape}")
#----------SUBJECT 2-------------

#X_train
#X_train_2 = get_windowed_feats(train_data2, 1000, 0.2,0.05)
#X_train_2 = np.nan_to_num(X_train_2)

#X_test
#X_test_2 = get_windowed_feats(test_data2, 1000, 0.2,0.05)
#X_test_2 = np.nan_to_num(X_test_2)

#y_train
#y_train_2 = get_windowed_dg(train_dg2, 25, 6, 2)

#R calc
#R2_train = create_R_matrix(X_train_2, 3)
#R2_test = create_R_matrix(X_test_2, 3)


#----------SUBJECT 3-------------

#X_train
#X_train_3 = get_windowed_feats(train_data3, 1000, 0.2,0.05)
#X_train_3 = np.nan_to_num(X_train_3)

#X_test
#X_test_3 = get_windowed_feats(test_data3, 1000, 0.2,0.05)
#X_test_3 = np.nan_to_num(X_test_3)

#y_train
#y_train_3 = get_windowed_dg(train_dg3, 25, 6, 2)

#R calc
#R3_train = create_R_matrix(X_train_3, 3)
#R3_test = create_R_matrix(X_test_3, 3)

# Neural Network

def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim = R1_train.shape[1], activation = 'relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation = 'relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation = 'relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation = 'relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))

    #choosing optimizer and performance metric
    model.compile(optimizer = 'adam', loss = Huber(delta=0.5))
    
    return model


# early stopping
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose=1, patience = 25)




formatted = lambda et : time.strftime("%H:%M:%S", time.gmtime(et))

#----------SUBJECT 1-------------

logging.info('Se aplica StandardScaler a R1 usando R1_train')
scaler1 = StandardScaler()
R1_train = scaler1.fit_transform(R1_train)

scaler_filename = "scaler1.save"
joblib.dump(scaler1, scaler_filename) 

R1_test = scaler1.transform(R1_test)

logging.info('Se aplica MinMaxScaler a y_train')
scaler_y = MinMaxScaler()
y_train_1 = scaler_y.fit_transform(y_train_1)


logging.info("Se hace split temporal dividiendo 90/10 train/val")
# Split temporal (antes de entrenar)
split_idx = int(0.9 * len(R1_train))
R1_train_part = R1_train[:split_idx]
y_train_part = y_train_1[:split_idx, :]


R1_val_part = R1_train[split_idx:]
y_val_part = y_train_1[split_idx:, :]



logging.debug(f"R1_train_part shape: {R1_train_part.shape}")
logging.debug(f"R1_val_part shape: {R1_val_part.shape}")

logging.debug(f"y_train_part shape: {y_train_part.shape}")
logging.debug(f"y_val_part shape: {y_val_part.shape}")




# finger 1
start_s1_f1 = time.time()
model_s1_f1 = build_model()
model_s1_f1.fit(R1_train_part, y_train_part[:,0],
                validation_data=(R1_val_part, y_val_part[:,0]),
                epochs=500, batch_size=32,
                callbacks=[early_stop], verbose=1)
end_s1_f1 = time.time()
model_s1_f1.save('models/model_s1f1.h5')
#predictions
test_pred_s1_f1 = model_s1_f1.predict(R1_test)

ets1f1 = -start_s1_f1+end_s1_f1
logging.info(f"Duración de entrenamiento para el dedo 1 del sujeto 1: {formatted(ets1f1)}")

#finger 2
start_s1_f2 = time.time()
model_s1_f2 = build_model()
model_s1_f2.fit(R1_train_part, y_train_part[:,1],
                validation_data=(R1_val_part, y_val_part[:,1]),
                epochs=500, batch_size=32,
                callbacks=[early_stop], verbose=1)
end_s1_f2 = time.time()
model_s1_f2.save('models/model_s1f2.h5')
#predictions
test_pred_s1_f2 = model_s1_f2.predict(R1_test)

ets1f2 = -start_s1_f2+end_s1_f2
logging.info(f"Duración de entrenamiento para el dedo 2 del sujeto 1: {formatted(ets1f2)}")
     
#finger 3
start_s1_f3 = time.time()
model_s1_f3 = build_model()
model_s1_f3.fit(R1_train_part, y_train_part[:,2],
                validation_data=(R1_val_part, y_val_part[:,2]),
                epochs=500, batch_size=64,
                callbacks=[early_stop], verbose=1)
end_s1_f3 = time.time()
model_s1_f3.save('models/model_s1f3.h5')
#predictions
test_pred_s1_f3 = model_s1_f3.predict(R1_test)

ets1f3 = -start_s1_f3+end_s1_f3
logging.info(f"Duración de entrenamiento para el dedo 3 del sujeto 1: {formatted(ets1f3)}")

#finger 4
start_s1_f4 = time.time()
model_s1_f4 = build_model()
model_s1_f4.fit(R1_train_part, y_train_part[:,3],
                validation_data=(R1_val_part, y_val_part[:,3]),
                epochs=500, batch_size=32,
                callbacks=[early_stop], verbose=1)
end_s1_f4 = time.time()
model_s1_f4.save('models/model_s1f4.h5')
#predictions
test_pred_s1_f4 = model_s1_f4.predict(R1_test)

ets1f4 = -start_s1_f4+end_s1_f4
logging.info(f"Duración de entrenamiento para el dedo 4 del sujeto 1: {formatted(ets1f4)}")

#finger 5
start_s1_f5 = time.time()
model_s1_f5 = build_model()
model_s1_f5.fit(R1_train_part, y_train_part[:,4],
                validation_data=(R1_val_part, y_val_part[:,4]),
                epochs=500, batch_size=64,
                callbacks=[early_stop], verbose=1)
end_s1_f5 = time.time()
model_s1_f5.save('models/model_s1f5.h5')
#predictions
test_pred_s1_f5 = model_s1_f5.predict(R1_test)

ets1f5 = end_s1_f5-start_s1_f5
logging.info(f"Duración de entrenamiento para el dedo 5 del sujeto 1: {formatted(ets1f5)}")

# Predictions

test_pred_s1_scaled = np.hstack((test_pred_s1_f1, test_pred_s1_f2, test_pred_s1_f3, test_pred_s1_f4, test_pred_s1_f5))
logging.debug(f"test_pred_s1_scaled shape: {test_pred_s1_scaled.shape}")
test_pred_s1 = scaler_y.inverse_transform(test_pred_s1_scaled)

xs = np.linspace(0,test_pred_s1.shape[0],test_data1.shape[0])

#interpolation
y1 = np.empty_like(test_pred_s1[:,0]) 
cs1 = [] 
for i in range(test_pred_s1.shape[1]):
    x1 = np.arange(test_pred_s1.shape[0])
    y1 = test_pred_s1[:,i]
    cs1.append(CubicSpline(x1, y1, bc_type = 'clamped'))

interp_pred1_nn = np.vstack((cs1[0](xs),cs1[1](xs), cs1[2](xs), cs1[3](xs), cs1[4](xs))).T

folder = 'Predictions'
try:
    os.mkdir(folder)
    logging.info(f"Carpeta {folder} Creada")
except:
    logging.info(f"Carpeta {folder} ya existía")
    
filename = folder + '/subj1_testpredictions.mat'
mat_dict = {'predicted_dg': interp_pred1_nn}
savemat(filename, mat_dict)

logging.info(f"Predicciones guardadas en {filename}")


END_TIME = time.time()

TOTAL_TIME = END_TIME-START_TIME

logging.info(f"Duración total del programa: {formatted(TOTAL_TIME)}")