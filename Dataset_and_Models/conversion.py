# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:37:02 2025

@author: israt
"""

import tensorflow as tf


model_s1_f1 = tf.keras.models.load_model('model_s1f1.h5')
model_s1_f2 = tf.keras.models.load_model('model_s1f2.h5')
model_s1_f3 = tf.keras.models.load_model('model_s1f3.h5')
model_s1_f4 = tf.keras.models.load_model('model_s1f4.h5')
model_s1_f5 = tf.keras.models.load_model('model_s1f5.h5')

converter1 = tf.lite.TFLiteConverter.from_keras_model(model_s1_f1)
tflite_model_s1f1 = converter1.convert()

# 3. Guardarlo como archivo .tflite
with open('model_s1f1.tflite', 'wb') as f:
    f.write(tflite_model_s1f1)

print("Conversión completa s1f1!")

converter2 = tf.lite.TFLiteConverter.from_keras_model(model_s1_f2)
tflite_model_s1f2 = converter2.convert()

# 3. Guardarlo como archivo .tflite
with open('model_s1f2.tflite', 'wb') as f:
    f.write(tflite_model_s1f2)

print("Conversión completa s1f2!")

converter3 = tf.lite.TFLiteConverter.from_keras_model(model_s1_f3)
tflite_model_s1f3 = converter3.convert()

# 3. Guardarlo como archivo .tflite
with open('model_s1f3.tflite', 'wb') as f:
    f.write(tflite_model_s1f3)

print("Conversión completa s1f3!")

converter4 = tf.lite.TFLiteConverter.from_keras_model(model_s1_f4)
tflite_model_s1f4 = converter4.convert()

# 3. Guardarlo como archivo .tflite
with open('model_s1f4.tflite', 'wb') as f:
    f.write(tflite_model_s1f4)

print("Conversión completa s1f4!")

converter5 = tf.lite.TFLiteConverter.from_keras_model(model_s1_f5)
tflite_model_s1f5 = converter5.convert()

# 3. Guardarlo como archivo .tflite
with open('model_s1f5.tflite', 'wb') as f:
    f.write(tflite_model_s1f5)

print("Conversión completa s1f5!")