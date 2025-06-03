import tensorflow as tf


model_path = "HelloWorld_Models\sine_model_perfect_int8.tflite" # Esto es suponiendo que se corre desde la base del repositorio

interprete = tf.lite.Interpreter(model_path=model_path)
interprete.allocate_tensors()

input_details = interprete.get_input_details()
output_details = interprete.get_output_details()
#tensor_details = interprete.get_tensor_details() # Todo

print("input_details = ", input_details ,'\n\n')
print("output_details = ", output_details, '\n\n' )
#print("tensor_details = ", tensor_details, '\n\n' )

