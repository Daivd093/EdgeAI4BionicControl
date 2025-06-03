import tensorflow as tf


model_path = "sine_model_perfect_int8.tflite"

interprete = tf.lite.Interpreter(model_path=model_path)
interprete.allocate_tensors()


