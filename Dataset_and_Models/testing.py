# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 06:34:58 2025

@author: israt
"""

import time
START_TIME = time.time()
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import logging

# Limpiar handlers anteriores
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='prueba_ecog.log',     # nombre del archivo de log
    level=logging.DEBUG,                    # nivel mínimo a registrar
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # o 'a' para agregar sin sobrescribir
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


logging.info('Cargando archivos')
groundtruth_subject1 = loadmat('sub1_testlabels.mat')
predicted_subject1 = loadmat('subj1_testpredictions.mat')


true_dg = groundtruth_subject1['test_dg']
nn_dg = predicted_subject1['predicted_dg']

assert len(true_dg) == len(nn_dg), logging.error("Largo de etiquetas reales y predichas es distinto.")


# Calcular r por dedo (excluyendo el dedo 4 → índice 3)
r_values = []
for i in range(5):
    if i == 3:
        continue  # saltar dedo 4
    r, _ = pearsonr(true_dg[:, i], nn_dg[:, i])
    logging.debug(f"Coeficiente de pearson para el dedo {i}: {r}")
    r_values.append(r)

# Promedio de los coeficientes de correlación
r_mean = np.mean(r_values)

print(f"r promedio (excluyendo dedo 4): {r_mean:.4f}")
logging.info(f"Performance: r = {r_mean}")



# Leaderboard 2008 (valores de r)
leaderboard = [
    ("Nanying Liang", 0.46),
    ("Remi Flamary", 0.42),
    ("Mathew Salvaris", 0.27),
    ("Florin Popescu", 0.10),
    ("Hyunjin Yoon", 0.05)
]


leaderboard.append(("Tú", r_mean))

leaderboard_ordenado = sorted(leaderboard, key=lambda x: x[1], reverse=True)

# Buscar tu posición
for idx, (nombre, r) in enumerate(leaderboard_ordenado, start=1):
    if nombre == "Tú":
        print(f"Con un r = {r_mean}, habrías quedado en el lugar #{idx}")
        break

print("\nLeaderboard actualizado:")
for idx, (nombre, r) in enumerate(leaderboard_ordenado, start=1):
    print(f"{idx}. {nombre} — r = {r}")