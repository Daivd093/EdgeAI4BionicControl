import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

model_file = "sine_model_perfect.h5"
model = load_model(model_file)

x_input = np.array([[0.0], [np.pi/2], [3.0], [4.0]], dtype=np.float32)
pred = model.predict(x_input)

for xi, yi in zip(x_input.flatten(), pred.flatten()):
    print(f"Predicción para x = {xi:.2f} → {yi:.6f}")

n_samples = 10000
x_values = np.linspace(start=-1,stop=2*np.pi+1,num=n_samples).reshape(-1, 1)
y_values = np.sin(x_values)
predictions = model.predict(x_values)
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_values,y_values,'b.',label='Actual')
plt.plot(x_values,predictions,'r.',label='Prediction')
plt.legend()
plt.show()

mse_model = 1/n_samples * (sum((y_values-predictions)**2))
print("Error cuadrático medio del modelo float32 local: ", mse_model)

# Para evaluar

