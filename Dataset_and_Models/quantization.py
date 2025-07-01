"""
Para hacer la cuantización requiero un dataset representativo, así que esta parte inicial es simplemente el mismo proceso para calcular R
Podría haber guardado R en el otro y ahorrarme esto, pero después lo arreglaré
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


logfile = logdir+'cuantizacion_ecog_info.log'
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
from scipy.io import loadmat
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
logging.info(f'Numpy {np.__version__}')

import tensorflow as tf
logging.info(f'Tensorflow {tf.__version__}')


import sklearn
logging.info(f'scikit-learn {sklearn.__version__}')




#Preprocess:
from utils import  get_windowed_feats, get_windowed_dg, create_R_matrix
    

# Train / Test split

#----------SUBJECT 2--------------
s = int(input("¿Para qué sujeto quiere hacer la cuantización? [1/2/3]: "))



filename = f"Dataset4_BCICIV/sub{s}_processed.npz"
logging.info(f"Esta versión va a intentar cargar {filename}, si no lo logra, va a hacer el preprocesamiento del dataset.")


try:
    with np.load(filename) as data:
        R_train = data["R_train"]
        R_test=data['R_test']
        y_train=data["y_train"]
        y_test=data["y_test"]
    logging.info(f"Dataset preprocesado cargado desde {filename} con éxito!")
    print("Si se pudo")
except:
    logging.info("No se pudo cargar dataset preprocesado")
    print("No se pudo")
    
    subject = loadmat(f'Dataset4_BCICIV/sub{s}_comp.mat')
    
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
    

# Generamos el dataset representativo

def representative_dataset_gen():
    for i in range(300):
        sample = R_train[i:i+1, :]  # batch de 1 muestra, las primeras 100 filas
        yield [sample.astype(np.float32)]



model_dir = "models/"
tflite_dir = "tflite/"

model_format = lambda s,f,e : f"model_s{s}f{f}.{e}"

for finger in range(5):

    filename_in = model_dir+model_format(s,finger+1,'h5')
    model = tf.keras.models.load_model(filename_in)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    filename_out = model_dir+tflite_dir+model_format(s,str(finger+1)+'_int8','tflite') 
    with open(filename_out, "wb") as f:
        f.write(tflite_model)
    
    logging.info(f"{filename_out} Creado con éxito!")