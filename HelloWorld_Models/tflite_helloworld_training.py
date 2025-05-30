# -*- coding: utf-8 -*-
"""
TFLITE Hello World

Fully Connected NN para predecir el seno

Ya que usaré STMCube.AI, no convertiré el .tflite en un .h

Quité el ruido de y_values y ahora guardo el .h5 y convierto a tf_lite en otro archivo
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

print('Numpy '+np.__version__)          # 1.26.4
print('Tensorflow '+tf.__version__)     # 2.10.0
print('Keras '+ tf.keras.__version__)   # 2.10.0

'''
import os
figdir = "figs/"
if not os.path.exists(figdir):
    os.makedirs(figdir)
'''


# Datos

nsamples = 1000
val_ratio = 0.2
test_ratio = 0.2
tflite_model_name = 'sine_model' #.tflite

#plt.figure()
np.random.seed(1138)
x_values = np.random.uniform(low=0, high=2*math.pi, size=nsamples)
#x_values = np.linspace(start=0,stop=2*math.pi,num=nsamples)
#plt.plot(x_values)
#plt.title("Gráfico de x_values")
#plt.show()
#plt.savefig(figdir+'x_values.png')
#plt.close()

plt.figure()
y_values = np.sin(x_values)+(0.1*np.random.randn(x_values.shape[0]))
plt.plot(x_values,y_values,'.')
plt.title("Gráfico de x_values vs y_values")
plt.show()
#plt.savefig(figdir+'x_vs_y.png')
#plt.close()


val_split = int(val_ratio * nsamples)
test_split = int(val_split + (test_ratio * nsamples))
x_val,x_test,x_train = np.split(x_values,[val_split,test_split])
y_val,y_test,y_train = np.split(y_values,[val_split,test_split])

assert (x_train.shape[0]+x_val.shape[0]+x_test.shape[0])==nsamples
plt.figure()
plt.plot(x_train,y_train,'r.',label='Train')
plt.plot(x_test,y_test,'g.',label='Test')
plt.plot(x_val,y_val,'b.',label='Validate')
plt.title("Gráfico de x_values vs y_values explicitando separación train/test/validate")
plt.legend()
plt.show()
#plt.savefig(figdir+'x_vs_y_train-test-val.png')
#plt.close() 

# Red neuronal
model = Sequential()
model.add(Dense(16,activation='relu',input_shape=(1,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='rmsprop',loss='mae',metrics=['mae'])

history = model.fit(x_train,y_train,epochs=500,batch_size=100,validation_data=(x_val,y_val))

model.save("sine_model_perfect.h5")

loss = history.history['loss']
val_los = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'ro',label='Training Loss')
plt.plot(epochs,val_los,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

predictions = model.predict(x_test)
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_test,y_test,'b.',label='Actual')
plt.plot(x_test,predictions,'r.',label='Prediction')
plt.legend()
plt.show()


x_input = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
pred = model.predict(x_input)

for xi, yi in zip(x_input.flatten(), pred.flatten()):
    print(f"Predicción para x = {xi:.1f} → {yi:.6f}")
