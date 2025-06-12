# -*- coding: utf-8 -*-
"""
TFLITE Hello World - Converting

Carga una red neuronal de keras en formato .h5 y lo convierte a modelos .tflite
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


model_name = "sine_model_perfect"

model = load_model(model_name+".h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open(model_name+'_32float'+'.tflite','wb').write(tflite_model)




### Versión INT8

def representative_dataset():
    for _ in range(100):
        x = np.random.uniform(0, 2*np.pi, size=(1, 1)).astype(np.float32)
        yield [x]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

with open(model_name+'_int8_justweights'+'.tflite', "wb") as f:
    f.write(tflite_model_quant)