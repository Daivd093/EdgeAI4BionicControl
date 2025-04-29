import numpy as np
import tensorflow as tf

# Carga el modelo .tflite
interpreter = tf.lite.Interpreter(model_path="sine_model.tflite")
interpreter.allocate_tensors()

# Entrada esperada
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Asigna el mismo valor que usaste en STM32
interpreter.set_tensor(input_details[0]['index'], np.array([[[[2.0]]]], dtype=np.float32))

# Ejecuta la inferencia
interpreter.invoke()

# Obtiene salida
output = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output)
