# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:16:14 2025

@author: david
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



logging.basicConfig(
    filename=logdir+'prueba_stm32_ecog.log',     # nombre del archivo de log
    level=logging.INFO,                    # nivel mínimo a registrar
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',  # o 'a' para agregar sin sobrescribir
    encoding='utf-8'
)

logging.info("======================Inicio======================")

import scipy
from scipy.io import loadmat
from scipy.stats import pearsonr
logging.info(f'Scipy {scipy.__version__}')

import numpy as np
import matplotlib.pyplot as plt
logging.info(f'Numpy {np.__version__}')
logging.info(f'MatPlotLib {plt.matplotlib.__version__}')

import tensorflow as tf

from utils import process_outputs

s = 2  #input("¿Qué sujeto quiere probar? [1/2/3] ")

try:
    logging.info('Cargando archivos')
    groundtruth = loadmat(f'Dataset4_BCICIV/sub{s}_testlabels.mat')
    predicted_pc = loadmat(f'Predictions/subj{s}_testpredictions.mat')
    predicted_stm32 = loadmat(f'Predictions/subj{s}_stm32_testpredictions_5f.mat')
    
except Exception as e:
    logging.error("Los archivos no están")
    logging.info(f"Error: {e}")

    
true_dg = groundtruth['test_dg']
nn_dg = predicted_pc['predicted_dg']
predicted_stm32_raw = predicted_stm32['unprocessed_predictions']


assert len(true_dg) == len(nn_dg), logging.error("Largo de etiquetas reales y predichas es distinto.")


nsamples = len(true_dg)
nn_stm_dg= process_outputs(predicted_stm32_raw, subject=s,expected_output_length=nsamples)


# Leaderboard 2008 (valores de r)
leaderboard = [
    ("Nanying Liang", 0.46),
    ("Remi Flamary", 0.42),
    ("Mathew Salvaris", 0.27),
    ("Florin Popescu", 0.10),
    ("Hyunjin Yoon", 0.05)
]



# Calcular r por dedo (excluyendo el dedo 4 → índice 3)
sources = ['Groundtruth', 'PC prediction','STM prediction']
r_values = []
r_mean = []
message = []
source0 = np.zeros(true_dg.shape)
source1 = np.zeros(true_dg.shape)
name = 'ERROR'
for n in range(len(sources)):
    
    r_values.append([])
    if n == 0:
        message.append(f"Pearson coeficient bewteen {sources[0]} and {sources[1]}")
        source0 = true_dg
        source1 = nn_dg
        name = 'Tú (PC)'
    elif n == 1:
        message.append(f"Pearson coeficient bewteen {sources[0]} and {sources[2]}")
        source0 = true_dg
        source1 = nn_stm_dg
        name = 'Tú (STM32)'
    elif n == 2:
        message.append(f"Pearson coeficient bewteen {sources[1]} and {sources[2]}") 
        source0 = nn_dg
        source1 = nn_stm_dg
    else:
        message.append("Pasó algo raro")
        logging.error("No deberías estar aquí")
    
    
    # Cálculo pearson
    n_dedos = source1.shape[1]
    for i in range(n_dedos):
        r, _ = pearsonr(source0[:, i], source1[:, i])
        logging.info(f"{message[n]} for the finger{i+1}: {r}")
        logging.debug(f"Error absoluto medio: {1/nsamples * sum(abs(source0[:,i]-source1[:,i]))}")
        if i != 3:
            r_values[n].append(r)


    r_mean.append(np.mean(r_values[n]))
    logging.info(f"Mean {message[n]} (excluding the 4th finger): {r_mean[n]:.4f}")
    
    if n <= 1:
        leaderboard.append((name, round(r_mean[n],3)))
        


leaderboard_ordenado = sorted(leaderboard, key=lambda x: x[1], reverse=True)

# Buscar tu posición
for idx, (nombre, r) in enumerate(leaderboard_ordenado, start=1):
    if "Tú" in nombre:
        print(f"Con un r = {r}, {nombre} habrías quedado en el lugar #{idx}")


print("\nLeaderboard actualizado:")
for idx, (nombre, r) in enumerate(leaderboard_ordenado, start=1):
    print(f"{idx}. {nombre} ————> r = {r}")
    
    
    
#Gráficos        
time = np.arange(nsamples) / 1000  # Las muestras están a 1kHz, con esto pasamos a segundos
for i in range(n_dedos):
    plt.figure(figsize=(10, 4))
    plt.plot(time, true_dg[:, i], label='Ground Truth', linewidth=1.5)
    plt.plot(time, nn_dg[:, i], label='Predicted by PC', linewidth=2, alpha=0.7)
    plt.plot(time, nn_dg[:, i], label='Predicted by STM', linewidth=0.7, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Finger #{i+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

