# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:37:02 2025

@author: Daivd093

Este archivo se encarga de la conversión de los modelos .h5 a .tflite, en su versión actual no realiza cuantización ni extrae un dataset representativo.

Una versión posterior probablemente fusionará conversión y cuantización.
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

logging.basicConfig(
    filename=logdir+'conversion_info.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filemode='w',
    encoding='utf-8'
)


import tensorflow as tf


finger_ids = [1, 2, 3, 4, 5]
subject_ids = [1,2,3]

model_dir = "models/"
tflite_dir = "tflite/"

try:
    os.mkdir(model_dir+tflite_dir)
except FileExistsError:
    pass  # ya existe, no pasa nada


model_format = lambda s,f,e : f"model_s{s}f{f}.{e}"

for s in subject_ids:
    for f in finger_ids:
        model = tf.keras.models.load_model(model_dir+model_format(s,f,'h5'))
        tflite_path = model_dir+tflite_dir+model_format(s,f,'tflite')
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as file:
            file.write(tflite_model)
        
        logging.info(f"Conversión completa para el dedo {f} del sujeto {s}")

