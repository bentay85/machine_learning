import tflite_runtime.interpreter as tflite
import timeit
from PIL import Image
import numpy as np

labels = ["chilli_crab", "curry_puff", "dim_sum","ice_kacang","kaya_toast","nasi_ayam","popiah","roti_prata","sambal_stingray","satay","tau_huay","wanton_noodle"]

img = Image.open("crab.jpeg").resize((224, 224))
img = np.array(img) * (1./255)

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

starttime = timeit.default_timer()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on image
input_shape = input_details[0]['shape']
input_data = np.float32(img[None,:,:,:]) 
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Time Taken:", timeit.default_timer() - starttime)

index=np.argmax(output_data,axis=1)[0]

print("Predicted Class: " + str(labels[index]))
