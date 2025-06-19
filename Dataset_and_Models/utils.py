# -*- coding: utf-8 -*-
"""
Este archivo contiene más que nada código de raimasen1729
https://github.com/raimasen1729/Finger-Flexion-Detection-Using-ECoG-Signal/tree/main
para hacer el preprocesado de los datos.

Este archivo existe para que los otros estén más limpios y simplemente importen las funciones que necesitan


Se hizo usando  Python 3.9.18
                Scipy 1.13.1
                Numpy 1.26.4
                MatPlotLib 3.9.2
                Tensorflow 2.10.0
                scikit-learn 1.6.1


"""
import logging
logger = logging.getLogger(__name__)



import scipy
from scipy import signal as sig
from scipy.signal import filtfilt
from scipy.signal import firwin
from scipy.signal import kaiserord
from copy import deepcopy
from scipy.fft import rfft, rfftfreq


import numpy as np

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
def Avg_Freq(x, fi, ff,fs=1000):
    #Convert to frequency domain:
    freq_sig = rfft(x)
    N = len(x)
    tdomain=rfftfreq(N,1/fs)  
    indices = np.where((tdomain>=fi) & (tdomain<=ff))[0]
    if len(indices) == 0:
        loggingg.warning(f"Sin datos en rango {fi}-{ff} Hz")
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


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    import sys
    print(f'Python {sys.version.split()[0]}')

    print(f'Scipy {scipy.__version__}')
    print(f'Numpy {np.__version__}')

